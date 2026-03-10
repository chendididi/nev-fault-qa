"""
知识图谱关系构建

基于实体抽取结果，构建以下关系：
- (FaultCode)-[:INDICATES]->(Component)     故障码指向相关零部件
- (Component)-[:BELONGS_TO]->(Subsystem)    零部件归属子系统
- (FaultCode)-[:CAUSED_BY]->(Component)     故障码可由某零部件引起
- (FaultCode)-[:HAS_SYMPTOM]->(Symptom)     故障码对应故障现象
- (Symptom)-[:REQUIRES_TOOL]->(Tool)        故障现象排查需要的工具
"""

from loguru import logger

from src.knowledge_graph.neo4j_client import Neo4jClient


# Cypher 语句模板
_MERGE_FAULT_CODE = """
MERGE (fc:FaultCode {code: $code})
ON CREATE SET fc.created_at = timestamp()
"""

_MERGE_COMPONENT = """
MERGE (c:Component {name: $name})
ON CREATE SET c.created_at = timestamp()
"""

_MERGE_SYMPTOM = """
MERGE (s:Symptom {description: $description})
ON CREATE SET s.created_at = timestamp()
"""

_MERGE_SUBSYSTEM = """
MERGE (sub:Subsystem {name: $name})
ON CREATE SET sub.created_at = timestamp()
"""

_MERGE_TOOL = """
MERGE (t:Tool {name: $name})
ON CREATE SET t.created_at = timestamp()
"""

_CREATE_INDICATES = """
MATCH (fc:FaultCode {code: $fault_code})
MATCH (c:Component {name: $component})
MERGE (fc)-[r:INDICATES]->(c)
ON CREATE SET r.weight = 1
ON MATCH SET r.weight = r.weight + 1
"""

_CREATE_HAS_SYMPTOM = """
MATCH (fc:FaultCode {code: $fault_code})
MATCH (s:Symptom {description: $symptom})
MERGE (fc)-[:HAS_SYMPTOM]->(s)
"""

_CREATE_BELONGS_TO = """
MATCH (c:Component {name: $component})
MATCH (sub:Subsystem {name: $subsystem})
MERGE (c)-[:BELONGS_TO]->(sub)
"""


def build_relations_from_chunk(chunk_entities: dict, neo4j: Neo4jClient) -> None:
    """
    根据单个 chunk 的实体抽取结果，在 Neo4j 中创建节点和关系。

    Args:
        chunk_entities: entity_extractor 的输出 dict
        neo4j: Neo4jClient 实例
    """
    fault_codes = chunk_entities.get("fault_codes", [])
    components = chunk_entities.get("components", [])
    symptoms = chunk_entities.get("symptoms", [])
    subsystems = chunk_entities.get("subsystems", [])
    tools = chunk_entities.get("tools", [])

    # 创建节点
    for fc in fault_codes:
        neo4j.run(_MERGE_FAULT_CODE, {"code": fc})
    for comp in components:
        neo4j.run(_MERGE_COMPONENT, {"name": comp})
    for sym in symptoms:
        neo4j.run(_MERGE_SYMPTOM, {"description": sym})
    for sub in subsystems:
        neo4j.run(_MERGE_SUBSYSTEM, {"name": sub})
    for tool in tools:
        neo4j.run(_MERGE_TOOL, {"name": tool})

    # 创建关系
    for fc in fault_codes:
        for comp in components:
            neo4j.run(_CREATE_INDICATES, {"fault_code": fc, "component": comp})
        for sym in symptoms:
            neo4j.run(_CREATE_HAS_SYMPTOM, {"fault_code": fc, "symptom": sym})

    for comp in components:
        for sub in subsystems:
            neo4j.run(_CREATE_BELONGS_TO, {"component": comp, "subsystem": sub})


def build_graph_from_chunks(chunks: list[dict], neo4j: Neo4jClient) -> None:
    """
    批量处理所有 chunk，构建完整知识图谱。

    Args:
        chunks: 含 entities 字段的 chunk 列表（entity_extractor 输出）
        neo4j: Neo4jClient 实例
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        entities = chunk.get("entities", {})
        if not entities:
            continue
        try:
            build_relations_from_chunk(entities, neo4j)
        except Exception as e:
            logger.warning(f"第 {i+1} 个 chunk 关系构建失败: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"知识图谱构建进度: {i+1}/{total}")

    logger.info("知识图谱构建完成")


def create_indexes(neo4j: Neo4jClient) -> None:
    """
    在 Neo4j 中创建约束和索引，提升查询性能。
    """
    statements = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (fc:FaultCode) REQUIRE fc.code IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.description IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sub:Subsystem) REQUIRE sub.name IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (fc:FaultCode) ON (fc.code)",
    ]
    for stmt in statements:
        neo4j.run(stmt)
    logger.info("Neo4j 索引和约束创建完成")
