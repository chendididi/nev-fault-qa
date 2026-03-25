"""
Neo4j 客户端封装

提供 Cypher 查询接口，支持：
- 按故障码查询相关零部件、子系统、故障现象
- 查询故障传播路径（因果链）
- 连接池管理
"""

from time import perf_counter
from typing import Any

from loguru import logger
from neo4j import GraphDatabase, Driver


class Neo4jClient:
    """Neo4j 连接与查询封装。"""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        trace_enabled: bool = False,
    ):
        """
        Args:
            uri: Bolt 连接地址，如 bolt://localhost:7687
            username: 用户名
            password: 密码
            database: 数据库名称
        """
        self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self._trace_enabled = trace_enabled
        logger.info(f"Neo4j 连接已建立: {uri}")

    def close(self) -> None:
        """关闭连接池。"""
        self._driver.close()

    def run(self, query: str, parameters: dict | None = None) -> list[dict[str, Any]]:
        """
        执行 Cypher 语句，返回结果列表。

        Args:
            query: Cypher 语句
            parameters: 查询参数

        Returns:
            List of dicts（每条记录为一个 dict）
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def query_fault_code(self, fault_code: str) -> dict[str, Any]:
        """
        查询故障码相关的完整信息（零部件、子系统、故障现象）。

        Args:
            fault_code: 故障码字符串，如 "P0300"

        Returns:
            dict with keys: fault_code, components, subsystems, symptoms
        """
        query = """
        MATCH (fc:FaultCode {code: $code})
        OPTIONAL MATCH (fc)-[:INDICATES]->(c:Component)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(sub:Subsystem)
        OPTIONAL MATCH (fc)-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN fc.code AS fault_code,
               collect(DISTINCT c.name) AS components,
               collect(DISTINCT sub.name) AS subsystems,
               collect(DISTINCT s.description) AS symptoms
        """
        started_at = perf_counter()
        results = self.run(query, {"code": fault_code.upper()})
        latency_ms = int((perf_counter() - started_at) * 1000)
        if self._trace_enabled:
            logger.bind(
                graph_query="fault_code",
                fault_code=fault_code.upper(),
                result_count=len(results),
                latency_ms=latency_ms,
            ).info("graph_query_completed")
        if not results:
            return {"fault_code": fault_code, "components": [], "subsystems": [], "symptoms": []}
        return results[0]

    def query_symptom_fault_codes(self, symptom_keywords: list[str]) -> list[dict]:
        """
        根据故障现象关键词查询可能的故障码。

        Args:
            symptom_keywords: 关键词列表，如 ["抖动", "无法启动"]

        Returns:
            List of dicts: [{fault_code, symptom, relevance_count}, ...]
        """
        # 用 CONTAINS 做模糊匹配
        conditions = " OR ".join(
            [f"s.description CONTAINS $kw{i}" for i in range(len(symptom_keywords))]
        )
        params = {f"kw{i}": kw for i, kw in enumerate(symptom_keywords)}
        query = f"""
        MATCH (fc:FaultCode)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE {conditions}
        RETURN fc.code AS fault_code, s.description AS symptom
        ORDER BY fc.code
        """
        started_at = perf_counter()
        results = self.run(query, params)
        latency_ms = int((perf_counter() - started_at) * 1000)
        if self._trace_enabled:
            logger.bind(
                graph_query="symptom_fault_codes",
                keyword_count=len(symptom_keywords),
                result_count=len(results),
                latency_ms=latency_ms,
            ).info("graph_query_completed")
        return results

    def query_causal_chain(self, fault_code: str, depth: int = 3) -> list[dict]:
        """
        查询故障码的因果传播链路（故障传播路径）。

        Args:
            fault_code: 起始故障码
            depth: 最大传播深度

        Returns:
            路径列表
        """
        query = """
        MATCH path = (fc:FaultCode {code: $code})-[:INDICATES|CAUSED_BY*1..$depth]->(end)
        RETURN [node IN nodes(path) | labels(node)[0] + ': ' + coalesce(node.code, node.name)] AS chain
        """
        started_at = perf_counter()
        results = self.run(query, {"code": fault_code.upper(), "depth": depth})
        latency_ms = int((perf_counter() - started_at) * 1000)
        if self._trace_enabled:
            logger.bind(
                graph_query="causal_chain",
                fault_code=fault_code.upper(),
                depth=depth,
                result_count=len(results),
                latency_ms=latency_ms,
            ).info("graph_query_completed")
        return results

    def get_stats(self) -> dict[str, int]:
        """返回图谱节点和关系统计。"""
        query = """
        MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt
        """
        results = self.run(query)
        return {r["label"]: r["cnt"] for r in results if r["label"]}

    def ping(self) -> bool:
        """轻量连接检查，用于 readiness。"""
        try:
            result = self.run("RETURN 1 AS ok")
        except Exception:
            return False
        return bool(result and result[0].get("ok") == 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
