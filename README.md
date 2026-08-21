<div align="center">

# ⚡ Agentic RAG 智能问答与知识治理平台

**基于 Java (Spring Boot) + Python (FastAPI) 双引擎架构的高性能 RAG 系统**  
*集成 Multi-Agent 协同调度 · 混合检索与重排序 · 三级记忆引擎 · 离线知识治理闭环 · 全链路 Trace 审计*

<p align="center">
  <img src="https://img.shields.io/badge/Java-17-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white" alt="Java 17" />
  <img src="https://img.shields.io/badge/Spring_Boot-3.2-6DB33F?style=for-the-badge&logo=spring-boot&logoColor=white" alt="Spring Boot 3.2" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Milvus-2.4+-00A1EA?style=for-the-badge&logo=zilliz&logoColor=white" alt="Milvus" />
  <img src="https://img.shields.io/badge/Redis-7.0-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis" />
  <img src="https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License" />
</p>

</div>

---

## 📖 项目简介

本项目是一个基于 **Java 后端 + Python AI 服务** 的 RAG 知识库系统。系统涵盖在线多智能体协同问答与离线知识库治理两大核心链路：在线侧通过混合检索、DAG 任务编排、多级记忆与语义缓存提升多跳问答准确度与响应速度；离线侧通过 Trace 审计日志采集未命中查询，定期巡检文档质量，形成知识补全与治理闭环。



### 核心功能模块



* **混合检索与重排序**：采用语义感知父子分块，结合 Milvus BM25 稀疏检索与 HNSW 稠密向量双路召回，通过 RRF 算法融合并使用 Cross-Encoder 进行语义重排序。

* **Multi-Agent 协同与自评**：面向多跳推理任务，基于 DAG 拓扑排序并行调度 Sub-Agent，通过 EventBus 共享上下文；Sub-Agent 结合 ReAct 范式进行检索充分性自评与改写重试。

* **三级记忆与二级语义缓存**：构建“工作记忆 → 长期记忆 (semantic/episodic/procedural) → 用户画像”三级体系，自动触发摘要沉淀；结合 Caffeine + Redis 二级缓存与 Milvus 语义相似度匹配加速高频请求。

* **全链路 Trace 审计**：在路由分发、LLM 调用、工具执行等节点统一埋点，跨 Java/Python 收集 I/O、耗时与 Token 消耗。

* **知识库离线治理**：基于问答与未命中日志聚类高频未覆盖 Query，巡检重复、低质、过期及冷门文档，反哺知识库维护。

---

## 🏗️ 系统整体架构

```mermaid
flowchart TB
    %% 全局样式定义
    classDef clientStyle fill:#EEF2F6,stroke:#94A3B8,stroke-width:1px,color:#1E293B;
    classDef javaStyle fill:#FFF7ED,stroke:#F97316,stroke-width:1.5px,color:#9A3412;
    classDef pythonStyle fill:#F5F3FF,stroke:#8B5CF6,stroke-width:1.5px,color:#5B21B6;
    classDef storeStyle fill:#ECFDF5,stroke:#10B981,stroke-width:1.5px,color:#065F46;
    classDef agentStyle fill:#FDF2F8,stroke:#EC4899,stroke-width:1px,color:#9D174D;
    classDef pipeStyle fill:#EFF6FF,stroke:#3B82F6,stroke-width:1px,color:#1E40AF;

    subgraph ClientLayer ["🖥️ 表现层 (React + SSE)"]
        UI_User["用户端 · 实时流式问答 / 引用溯源"]
        UI_Admin["管理端 · 知识治理工作台 / 巡检报表"]
    end
    class UI_User,UI_Admin clientStyle

    subgraph JavaGateway ["☕ 业务中台 (Spring Boot 3.2)"]
        direction TB
        API_Gateway["API 路由分发 / JWT 权限鉴权"]
        SSE_Engine["SSE 流式推送引擎"]
        L1_Cache["Caffeine 本地热点缓存"]
        Trace_Filter["Trace 链路拦截 (TraceId 注入)"]
        Biz_Service["业务管理 / 审计持久化"]
    end
    class API_Gateway,SSE_Engine,L1_Cache,Trace_Filter,Biz_Service javaStyle

    subgraph PythonCore ["🐍 AI 算力与智能体服务 (FastAPI)"]
        direction TB
        Orchestrator["DAG 拓扑任务编排调度器"]
        EventBus[("EventBus 事件总线 (状态同步)")]
        
        subgraph SubAgents ["🤖 Multi-Agent 协同矩阵"]
            A_Router["Intent Router"]
            A_Retrieval["Retrieval Agent"]
            A_Analysis["Analysis Agent"]
            A_Eval["Self-Correction Agent"]
        end
        class A_Router,A_Retrieval,A_Analysis,A_Eval agentStyle

        subgraph PipelineEngine ["🔍 检索与重排管线"]
            Chunk_Engine["语义父子分块解析"]
            Hybrid_Search["双路召回 (BM25 + HNSW)"]
            RRF_Merge["RRF 倒数排名融合"]
            Cross_Rerank["Cross-Encoder 重排序"]
        end
        class Chunk_Engine,Hybrid_Search,RRF_Merge,Cross_Rerank pipeStyle

        Memory_Engine["三级记忆引擎 (Working / Episodic / Profile)"]
        Governance_Engine["知识库离线治理 (聚类分析 / 异常巡检)"]
    end
    class Orchestrator,EventBus,Memory_Engine,Governance_Engine pythonStyle

    subgraph DataLayer ["💾 混合存储基础设施"]
        DB_MySQL[("MySQL 8.0<br/>业务元数据 / 审计日志")]
        DB_Redis[("Redis 7.0<br/>分布式二级缓存 / 会话")]
        DB_Milvus[("Milvus 2.4+<br/>稠密/稀疏向量 / 语义缓存")]
    end
    class DB_MySQL,DB_Redis,DB_Milvus storeStyle

    %% 交互连线
    UI_User -->|HTTP / SSE| API_Gateway
    UI_Admin -->|REST API| API_Gateway
    
    API_Gateway --> L1_Cache
    L1_Cache -.->|Cache Miss| DB_Redis
    API_Gateway --> SSE_Engine
    API_Gateway -->|RPC / 异步转发| Orchestrator
    Trace_Filter -.->|跨语言 TraceId 透传| Orchestrator

    Orchestrator --> EventBus
    EventBus <--> SubAgents
    
    A_Retrieval --> PipelineEngine
    A_Analysis --> Memory_Engine
    A_Eval -->|充分性不足| A_Retrieval

    PipelineEngine <--> DB_Milvus
    Memory_Engine <--> DB_Redis & DB_Milvus
    Biz_Service --> DB_MySQL
    Governance_Engine --> DB_MySQL & DB_Milvus
```

---

## ⚙️ 核心技术实现

### 1. 混合检索与上下文重构管线

采用“**小切块检索召回，大切块提供上下文**”策略，解决细粒度检索精确度与大模型上下文完整度之间的冲突。

```mermaid
flowchart LR
    classDef default fill:#F8FAFC,stroke:#64748B,stroke-width:1px,color:#0F172A;
    classDef highlight fill:#EFF6FF,stroke:#3B82F6,stroke-width:1.5px,color:#1E40AF;
    classDef success fill:#ECFDF5,stroke:#10B981,stroke-width:1.5px,color:#065F46;

    Doc["📄 原始输入文档"] --> Split{"✂️ 语义感知分块"}
    
    Split -->|Macro Chunk| Parent["📦 父 Chunk (800~1500 tokens)<br/>保留完整业务段落与上下文"]
    Split -->|Micro Chunk| Child["🧩 子 Chunk (200~300 tokens)<br/>细粒度语义片段"]

    Child --> BM25["🔤 BM25 稀疏索引 (关键词)"]
    Child --> Dense["📐 HNSW 稠密向量索引 (语义)"]

    BM25 --> RRF{"⚖️ RRF 排名融合"}
    Dense --> RRF

    RRF --> TopK["Top-K 候选子块"]
    TopK --> Rerank["🎯 Cross-Encoder 深度重排"]
    
    Rerank --> Expand["🔄 上下文对齐 (回溯父 Chunk)"]
    Parent -.->|父子关系回填| Expand
    
    Expand --> LLM["🤖 LLM 上下文生成"]

    class Split,RRF highlight
    class Parent,Expand,LLM success
```

### 2. DAG 驱动的 Multi-Agent 协同与自评回路

对于复合型复杂多跳提问，系统通过拓扑排序自动分解任务依赖，支持并行分支与检索质量自愈。

```mermaid
flowchart TD
    classDef normal fill:#F8FAFC,stroke:#64748B,stroke-width:1px,color:#0F172A;
    classDef condition fill:#FEF9C3,stroke:#CA8A04,stroke-width:1.5px,color:#854D0E;
    classDef accent fill:#F3E8FF,stroke:#9333EA,stroke-width:1.5px,color:#581C87;
    classDef complete fill:#ECFDF5,stroke:#059669,stroke-width:1.5px,color:#065F46;

    Query["🗣️ 用户复合提问"] --> Planner["📋 DAG 任务规划器"]

    subgraph DAG_Execution ["多智能体并行调度"]
        Planner --> SubA["🤖 智能体 A<br/>子任务拆解 1"]
        Planner --> SubB["🤖 智能体 B<br/>子任务拆解 2"]
        
        SubA & SubB -->|发布结果| Bus[("📦 EventBus 状态共享")]
        Bus -->|依赖就绪通知| SubC["🤖 智能体 C<br/>汇总推理分析"]
    end
    class SubA,SubB,SubC accent

    SubC --> Eval{"⚡ 检索充分性自评"}
    class Eval condition

    Eval -- "❌ 信息不充分 (Score < Threshold)" --> Rewrite["🔄 Query 改写 / 扩展重试"]
    Rewrite -->|重新检索| SubA

    Eval -- "✅ 信息充分" --> Gen["📝 结构化响应生成"]
    Gen --> SSE["📡 SSE 流式吐出至前端"]
    class Gen,SSE complete
```

### 3. 三级记忆与二级语义缓存机制

* **三级记忆架构**：
  * **工作记忆 (Working Memory)**：基于当前会话上下文窗口，管理轮次对话。
  * **长期记忆 (Long-term Memory)**：分类沉淀为 `Semantic`（事实知识）、`Episodic`（交互事件）与 `Procedural`（操作规程），由后台异步 LLM 自动提取并向量化入库。
  * **用户画像 (User Profile)**：维护业务偏好与常用领域标签，个性化调整 Prompt 权重。
* **二级语义缓存加速**：
  * **L1 本地缓存**：`Caffeine` 提供微秒级单机热点拦截。
  * **L2 分布式语义缓存**：`Redis + Milvus` 协同。输入 Query 首先进入 Milvus 执行向量余弦相似度匹配，若高于预设阈值（如 `0.95`）则判定语义命中，直接返回缓存结果并延长热点 Key 的 TTL。

### 4. 闭环知识库离线治理体系

```mermaid
flowchart LR
    classDef stream fill:#EFF6FF,stroke:#3B82F6,stroke-width:1px,color:#1D4ED8;
    classDef batch fill:#FFF7ED,stroke:#EA580C,stroke-width:1px,color:#9A3412;
    classDef fix fill:#ECFDF5,stroke:#10B981,stroke-width:1px,color:#065F46;

    subgraph Online ["🔵 在线服务"]
        Q["用户查询"] --> Miss["未命中 / 低置信度 / 负反馈"]
    end
    class Q,Miss stream

    subgraph Offline ["🟠 离线治理"]
        Miss --> Log[("Trace 审计库")]
        Log --> Cluster["HDBSCAN 未覆盖 Query 聚类"]
        Cluster --> Gap["📊 输出高频知识缺口报表"]
        
        Inspect["定时巡检器"] --> Scan["扫描重复/低质/过期文档"]
    end
    class Log,Cluster,Gap,Inspect,Scan batch

    subgraph Action ["🟢 知识反哺"]
        Gap & Scan --> Workbench["🛠️ 知识治理工作台"]
        Workbench --> Update["✏️ 定向补充 / 修正废弃文档"]
        Update --> Sync["🔄 向量库自动增量重构"]
    end
    class Workbench,Update,Sync fix

    Sync -.->|持续优化召回率| Online
```

---

## 🛠️ 技术栈清单

| 层次 / 模块 | 技术选型 | 核心作用说明 |
| :--- | :--- | :--- |
| **业务中台** | `Java 17` / `Spring Boot 3.2` / `MyBatis-Plus` | 核心业务路由、高并发鉴权、审计持久化 |
| **流式通信** | `Spring WebFlux` (SSE) | 生产级 Server-Sent Events 流式响应与心跳保活 |
| **AI 引擎** | `Python 3.10+` / `FastAPI` / `Pydantic` | 异步 AI 服务、DAG 编排、多智能体协同 |
| **向量数据库** | `Milvus 2.4+` | BM25 稀疏检索、HNSW 稠密检索与语义缓存索引 |
| **持久化存储** | `MySQL 8.0` | 业务元数据、Trace 全链路审计日志持久化 |
| **多级缓存** | `Caffeine` + `Redis 7.0` | 本地微秒级缓存 + 分布式高可用会话与缓存 |
| **大模型 / 向量** | `DashScope (Qwen-Plus)` / `Cross-Encoder` | 复杂多跳推理生成、高维文本向量化与精准重排 |
| **质量评估** | `RAGAS` | 针对上下文相关性、忠实度与召回率的自动化评测 |

---

## 🚀 快速启动

### 1. 基础环境依赖

请确保本地或服务器已安装并启动以下中间件：
* **MySQL 8.0**（默认端口 `3306`）
* **Redis 7.0**（默认端口 `6379`）
* **Milvus 2.4+**（默认端口 `19530`）

### 2. 数据库与元数据初始化

```bash
mysql -u root -p -e "CREATE DATABASE ai_knowledge_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -u root -p ai_knowledge_db < sql/init.sql
```

### 3. 启动 Python AI 服务 (Port: 8000)

```bash
cd ai-service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 DASHSCOPE_API_KEY 与 Milvus/Redis 连接信息

python main.py
```

### 4. 启动 Java 业务中台 (Port: 8080)

```bash
cd backend-service
# 修改 src/main/resources/application.yml 中的数据源配置
mvn clean package -DskipTests
java -jar target/ai-knowledge-system-*.jar
```

### 5. 启动前端交互界面 (Port: 3000)

```bash
cd frontend
npm install
npm run dev
```

### 平台入口

* **用户端对话界面**：`http://localhost:3000/login`（支持手机验证码登录）
* **知识治理管理端**：`http://localhost:3000/admin/login`（默认账号：`admin` / `admin123`）

---

## 📂 项目工程结构

```text
.
├── backend-service/                # Java 业务中台工程 (Spring Boot)
│   ├── src/main/java/com/rag/
│   │   ├── controller/             # RESTful API 与 SSE 流式控制器
│   │   ├── service/                # 业务逻辑与定时巡检任务
│   │   ├── cache/                  # Caffeine + Redis 多级缓存封装
│   │   └── trace/                  # 跨语言 TraceId 拦截与审计存储
│   └── src/main/resources/         # 基础配置与 MyBatis Mapper XML
├── ai-service/                     # Python AI 核心服务 (FastAPI)
│   ├── core/
│   │   ├── chunking/               # 语义感知父子分块实现
│   │   ├── retrieval/              # 混合检索 (BM25+HNSW) 与重排
│   │   └── evaluation/             # RAGAS 指标自动化评估
│   ├── agents/
│   │   ├── orchestrator.py         # DAG 拓扑执行引擎
│   │   ├── event_bus.py            # Agent 间状态共享总线
│   │   └── sub_agents/             # 细分 Agent 逻辑实现
│   ├── memory/                     # 三级记忆管理 (Working/Episodic/Profile)
│   └── governance/                 # Query 聚类分析与知识巡检
├── sql/                            # 数据库建表与初始化脚本
└── README.md
```

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 协议开源