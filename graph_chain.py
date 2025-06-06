from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from rapidfuzz import process, fuzz
from neo4j import GraphDatabase
from py2neo import Graph as Py2NeoGraph  # 加入py2neo


# ---------- 实体结构定义 ----------
class QuestionEntities(BaseModel):
    Authors: Optional[List[str]] = Field(default=[], description="Author names")
    Papers: Optional[List[str]] = Field(default=[], description="Full titles of papers in the area of optics")
    Years: Optional[List[int]] = Field(default=[], description="Years")
    Sources: Optional[List[str]] = Field(default=[], description="Journal or conference names")
    Keywords: Optional[List[str]] = Field(default=[], description="Keywords excluding 'optical frequency comb'")


class CypherQuery(BaseModel):
    query: str


# ---------- 实体标准化工具 ----------
def build_lowercase_mapping(entity_list: List[str]) -> Dict[str, str]:
    return {name.lower(): name for name in entity_list if isinstance(name, str)}


def correct_entity(entity: str, entity_type: str, cache: Dict[str, List[str]], threshold: float = 90.0) -> Optional[str]:
    if not isinstance(entity, str):
        return entity if entity_type == "Years" else None
        
    if entity_type not in cache:
        return None

    candidates = cache[entity_type]
    lower_map = build_lowercase_mapping(candidates)
    entity_lower = entity.strip().lower()

    if entity_lower in lower_map:
        return lower_map[entity_lower]

    best_match = process.extractOne(entity, candidates, scorer=fuzz.token_sort_ratio)
    if best_match:
        match, score, _ = best_match
        if score >= threshold:
            return match

    return None


def correct_all_entities(extracted: Dict[str, List], entity_cache: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        entity_type: list(filter(None, [
            correct_entity(e, entity_type, entity_cache)
            for e in entity_list
        ]))
        for entity_type, entity_list in extracted.items()
    }


# ---------- 获取数据库中的实体 ----------
def get_node_names(driver, label: str, attribute: str) -> List[str]:
    query = f"Match (n:{label}) RETURN DISTINCT n.{attribute} AS name"
    with driver.session() as session:
        result = session.run(query)
        return [record['name'] for record in result if record['name'] is not None]


def build_entities_dict(driver) -> Dict[str, List[str]]:
    return {
        "Authors": get_node_names(driver, "Author", "name"),
        "Papers": get_node_names(driver, "Paper", "title"),
        "Years": get_node_names(driver, "Year", "year"),
        "Sources": get_node_names(driver, "Source", "name"),
        "Keywords": get_node_names(driver, "Keyword", "name"),
    }


def build_graph_qa_chain(llm, graph, entity_cache):
    # Step 1: 实体抽取链构建（使用 PydanticOutputParser）
    parser = PydanticOutputParser(pydantic_object=QuestionEntities)

    entity_prompt = PromptTemplate(
        template="""
        You are extracting graph-related entities from user questions. 
        The topic is always optical frequency comb, so exclude that as a keyword. But more complex words like 'optical frequency comb generation' is a keyword.
        Please distinguish between the Papers and the keywords. 
        For example: Nonlinear conversion efficiency in Kerr frequency comb generation is a whole title of a paper.
        Do not split it into several keywords.
        
        Extract the following entities from the question:
        - Authors: Names of authors or researchers.
        - Papers: Titles of papers or articles.
        - Years: Publication years or dates.
        - Sources: Journals, conferences, or other publication sources.
        - Keywords: Important terms or phrases in the question that related to the topic.
        
        If a specific entity type is not present in the question, return an empty list for that type.
        
        Format your response as a JSON that matches this schema:
        {format_instructions}
        
        Question: {question}
        """,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    entity_chain = entity_prompt | llm | parser

    cypher_prompt_text = '''
    You are a Cypher generation expert, responsible for translating natural language problems into queries against the following Neo4j graph databases,
    all the queries are about optical frequency comb. 
    
    [Schema Definition]
    
    Node type:
    
    - Paper: attributes include title (title of the paper),  Abstract (Abstract of the paper), 
    Citations(citation count of the paper, it can be a standard to measure the importance of the paper)
    
    - Author: The author attribute includes name and author_ids.
    
    - Year: attribute year (integer),
    
    - Source: attribute name (conference/journal name), 
    
    -Keyword: attribute name (keywords), all the papers are about optical frequency comb. 
    So if the user ask questions about optical frequency comb, DO NOT use it as a keyword.
    
    
    Relationship type:
    
    - (Author)-[:WROTE]->(Paper) : The author wrote the paper
    
    - (Paper)-[:CITES]->(Paper) : The paper cites other papers
    
    - (Paper)-[:PUBLISHED_IN_YEAR]->(Year) : year of publication
    
    - (Paper)-[:PUBLISHED_IN_SOURCE]->(Source) : publication source
    
    - (Paper)-[:KEYWORD]->(Keyword) : keyword of the paper,all the papers are about optical frequency comb. 
    So if the user ask questions about optical frequency comb, DO NOT use it as a keyword.
    
    Please generate the Cypher code based on the given Node and Relationship type.
    
    [Example]
    
    Question: What is the content of Nonlinear conversion efficiency in Kerr frequency comb generation?
    
    Cypher: MATCH (p:Paper {{title: "Nonlinear conversion efficiency in Kerr frequency comb generation"}}) 
    RETURN p.Abstract AS abstract 
    
    Question: What is the development of Optical frequency comb generation since 2018?
    
    Cypher: 
    MATCH (p:Paper)-[:PUBLISHED_IN_YEAR]->(y:Year), (p)-[:KEYWORD]->(kw:Keyword)
    WHERE kw.name = 'Optical frequency comb generation' 
    AND y.year >= 2018
    WITH p, y
    ORDER BY p.Citations DESC
    LIMIT 5
    RETURN p.title AS title, y.year AS year, p.Abstract AS Abstract
    
    Turn the user's question into a valid Cypher query according to the above rules.
    Generate Cypher query fully based on the given entities extracted from the user question.
    The result name MUST be the same as the node or attributes name(such as p.Author as Author).
    Confirm only the Cypher code output without interpretation.
    
    User question:{question}.'''

    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", cypher_prompt_text),
        ("human", "Given these entities: {entities}, generate a Cypher query.")
    ])
    cypher_chain = cypher_prompt | llm.with_structured_output(CypherQuery)

    # ---------- 最终回答用的Prompt ----------
    answer_template = '''
    You are an expert of optical frequency comb. Your task is to answer questions related to the topic.
    The questions are all about titles, authors and other entities of literatures in the area of optical frequency comb.
    I will provide you some contexts to help you answer the question.
    The contexts are retrieved using the question from a neo4j graph database using cypher.

    Cypher: {cypher}
    The Retrieved Contexts of The Question: {context}
    Question: {question}
    '''
    answer_prompt = ChatPromptTemplate.from_template(answer_template)

    # ---------- Pipeline 流程 ----------
    def run_pipeline(input: Dict) -> Dict:
        question = input["question"]

        # Step 1: 实体抽取
        extracted = entity_chain.invoke({"question": question})
        print(extracted)

        # Step 2: 实体修正
        corrected_entities = correct_all_entities(extracted.dict(), entity_cache)
        print(corrected_entities)

        if (len(corrected_entities['Keywords']) < len(extracted.dict()['Keywords'])) and \
            all(not value for key, value in corrected_entities.items() if key != 'Keywords'):
            print("keyword error")
            return None


        # Step 3: 生成 Cypher
        cypher = cypher_chain.invoke({
            "entities": corrected_entities,
            "question": question
        })
        print(cypher)

        # Step 4: 执行 Cypher
        context_result = graph.run(cypher.query).data()

        if not context_result:
            print("context error")
            return None

        print(context_result)

        # Step 5: 用上下文+问题组装回答
        formatted_prompt = answer_prompt.format_messages(
            cypher = cypher.query,
            question=question,
            context=context_result
        )
        response = llm.invoke(formatted_prompt)

        return {
            "question": question,
            "extract": extracted,
            "entities": QuestionEntities(**corrected_entities),
            "cypher": cypher.query,
            "context": context_result,
            "answer": response.content
        }

    return RunnableLambda(run_pipeline)
