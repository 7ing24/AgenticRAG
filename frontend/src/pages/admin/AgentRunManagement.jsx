import { useState, useEffect } from 'react';
import './AdminDashboard.css';

const statusColors = {
  PENDING: '#d9d9d9',
  RUNNING: '#1890ff',
  COMPLETED: '#52c41a',
  FAILED: '#ff4d4f',
  INTERRUPTED: '#faad14',
  WAITING: '#13c2c2'
};

const statusLabels = {
  PENDING: '待执行',
  RUNNING: '执行中',
  COMPLETED: '已完成',
  FAILED: '失败',
  INTERRUPTED: '已中断',
  WAITING: '等待中'
};

const agentTypeLabels = {
  chitchat: 'Chitchat',
  knowledge_qa: 'Knowledge Q&A',
  agent_orchestrator: 'Agent Orchestrator',
  multi_agent_orchestrator: 'Agent Orchestrator',  // 旧数据兼容
  react_agent: 'ReAct Worker',
  admin_copilot: 'Admin Copilot',
  ops_agent: 'Operation Analysis',
};

const stepTypeLabels = {
  INTENT_RECOGNITION: '意图识别',
  QUESTION_CLASSIFICATION: '问题分类',
  QUESTION_REWRITE: '问题改写',
  KNOWLEDGE_SEARCH: '知识检索',
  RESULT_EVALUATION: '结果评估',
  ANSWER_GENERATION: '答案生成',
  MEMORY_WRITE: '记忆写入',
  TOOL_CALL: '工具调用',
  planning: '规划',
  synthesis: '综合汇总',
  llm_generation: 'LLM生成',
  intent_parsing: '意图解析',
  data_analysis: '数据分析',
};

function normalizeStatus(s) {
  if (!s) return 'UNKNOWN';
  return s.toUpperCase();
}

function truncate(s, max) {
  if (!s) return '-';
  return s.length > max ? s.substring(0, max) + '...' : s;
}

export default function AgentRunManagement() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [traceRuns, setTraceRuns] = useState([]);
  const [selectedTraceId, setSelectedTraceId] = useState(null);
  const [detailError, setDetailError] = useState('');

  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  const [filters, setFilters] = useState({ userId: '', status: '', dateStart: '', dateEnd: '' });

  const fetchList = (page, size) => {
    setLoading(true);
    const p = page || pageNum;
    const s = size || pageSize;
    let url = `/api/agent-run/list?pageNum=${p}&pageSize=${s}`;
    if (filters.userId) url += `&userId=${filters.userId}`;
    if (filters.status) url += `&status=${filters.status}`;
    if (filters.dateStart) url += `&startTime=${filters.dateStart}T00:00:00`;
    if (filters.dateEnd) url += `&endTime=${filters.dateEnd}T23:59:59`;

    const token = localStorage.getItem('adminToken');
    fetch(url, { headers: { 'Content-Type': 'application/json', 'Authorization': token ? `Bearer ${token}` : '' } })
      .then(r => r.json())
      .then(result => {
        if (result.code === 200) {
          setData(result.data.records || []);
          setTotal(result.data.total || 0);
        }
      })
      .catch(e => console.error('获取列表失败:', e))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchList(1); }, []);

  const handleSearch = () => { setPageNum(1); fetchList(1); };
  const handleReset = () => { setFilters({ userId: '', status: '', dateStart: '', dateEnd: '' }); setPageNum(1); fetchList(1); };

  const goToPage = (page) => { if (page < 1 || page > totalPages) return; setPageNum(page); fetchList(page); };
  const changePageSize = (size) => { setPageSize(size); setPageNum(1); fetchList(1, size); };

  const openTraceDetail = async (item) => {
    const traceId = item.traceId;
    if (!traceId) {
      setDetailError('该记录缺少 traceId，无法查看链路（可能是旧数据）');
      setTraceRuns([]);
      setSelectedTraceId(null);
      setDetailVisible(true);
      return;
    }
    setDetailError('');
    setSelectedTraceId(traceId);
    setDetailLoading(true);
    setDetailVisible(true);
    try {
      const token = localStorage.getItem('adminToken');
      const res = await fetch(`/api/agent-run/trace/${encodeURIComponent(traceId)}`, {
        headers: { 'Content-Type': 'application/json', 'Authorization': token ? `Bearer ${token}` : '' }
      });
      const result = await res.json();
      if (result.code === 200) {
        setTraceRuns(result.data || []);
      } else {
        setDetailError(result.message || '获取链路详情失败');
        setTraceRuns([]);
      }
    } catch (e) {
      console.error('获取Trace详情失败:', e);
      setDetailError('网络请求失败：' + e.message);
      setTraceRuns([]);
    } finally {
      setDetailLoading(false);
    }
  };

  const formatTime = (time) => {
    if (!time) return '-';
    return new Date(time).toLocaleString('zh-CN');
  };

  const totalPages = Math.ceil(total / pageSize);

  // 构建 trace detail 的树形结构
  const buildTraceTree = (runs) => {
    const rootRuns = runs.filter(r => !r.parentRunId);
    const childRuns = runs.filter(r => r.parentRunId);
    return rootRuns.map(root => ({
      ...root,
      children: childRuns.filter(c => c.parentRunId === root.runId)
    }));
  };

  return (
    <div className="admin-panel">
      <div className="page-header">
        <h1>Agent 执行记录</h1>
        <p>查看所有 Agent 的执行链路和步骤详情</p>
      </div>


      {/* 筛选 */}
      <div className="search-bar">
        <div className="search-row">
          <div className="search-item">
            <label>用户ID:</label>
            <input value={filters.userId} onChange={e => setFilters({...filters, userId: e.target.value})} placeholder="输入用户ID" />
          </div>
          <div className="search-item">
            <label>状态:</label>
            <select value={filters.status} onChange={e => setFilters({...filters, status: e.target.value})}>
              <option value="">全部</option>
              {Object.entries(statusLabels).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
          <div className="search-item">
            <label>日期:</label>
            <input type="date" value={filters.dateStart} onChange={e => setFilters({...filters, dateStart: e.target.value})} />
            <span>~</span>
            <input type="date" value={filters.dateEnd} onChange={e => setFilters({...filters, dateEnd: e.target.value})} />
          </div>
          <div className="search-actions">
            <button className="btn btn-primary" onClick={handleSearch}>搜索</button>
            <button className="btn btn-default" onClick={handleReset}>重置</button>
          </div>
        </div>
      </div>

      {/* 主表 */}
      <div className="table-container">
        <table className="admin-table">
          <thead>
            <tr>
              <th>用户ID</th>
              <th>会话ID</th>
              <th>问题</th>
              <th>类型</th>
              <th>状态</th>
              <th>时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan="8" className="loading">加载中...</td></tr>
            ) : data.length === 0 ? (
              <tr><td colSpan="8" className="empty">暂无数据</td></tr>
            ) : (
              data.map(item => {
                const status = normalizeStatus(item.status);
                return (
                  <tr key={item.runId}>
                    <td>{item.userId || '-'}</td>
                    <td title={item.conversationId}>{truncate(item.conversationId, 10)}</td>
                    <td className="ellipsis" title={item.input}>{truncate(item.input, 25)}</td>
                    <td>{item.agentType ? (agentTypeLabels[item.agentType] || item.agentType) : '-'}</td>
                    <td>
                      <span className="status-badge" style={{
                        backgroundColor: (statusColors[status] || '#d9d9d9') + '20',
                        color: statusColors[status] || '#666'
                      }}>{statusLabels[status] || status}</span>
                    </td>
                    <td>{formatTime(item.startTime)}</td>
                    <td>
                      <button className="action-btn edit" onClick={() => openTraceDetail(item)}>查看详情</button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* 分页 */}
      {total > 0 && (
        <div className="pagination">
          <div className="pagination-info">
            共 {total} 条，第 {pageNum}/{totalPages} 页
            <select value={pageSize} onChange={e => changePageSize(Number(e.target.value))}>
              <option value={10}>10条/页</option>
              <option value={20}>20条/页</option>
              <option value={50}>50条/页</option>
            </select>
          </div>
          <div className="pagination-controls">
            <button disabled={pageNum <= 1} onClick={() => goToPage(1)}>首页</button>
            <button disabled={pageNum <= 1} onClick={() => goToPage(pageNum - 1)}>上一页</button>
            {(() => {
              const pages = [];
              const start = Math.max(1, pageNum - 2);
              const end = Math.min(totalPages, pageNum + 2);
              for (let i = start; i <= end; i++) {
                pages.push(<button key={i} className={i === pageNum ? 'btn-primary' : ''} onClick={() => goToPage(i)}>{i}</button>);
              }
              return pages;
            })()}
            <button disabled={pageNum >= totalPages} onClick={() => goToPage(pageNum + 1)}>下一页</button>
            <button disabled={pageNum >= totalPages} onClick={() => goToPage(totalPages)}>末页</button>
          </div>
        </div>
      )}

      {/* Trace 详情弹窗 */}
      {detailVisible && (
        <div className="modal-overlay" onClick={() => setDetailVisible(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 960, width: '92%' }}>
            <div className="modal-header">
              <h3>Trace 链路 — {selectedTraceId}</h3>
              <button className="modal-close" onClick={() => setDetailVisible(false)}>×</button>
            </div>
            <div className="modal-body">
              <div className="detail-content">
                {detailError && (
                  <div style={{ padding: 16, background: '#fff2f0', borderRadius: 6, color: '#ff4d4f', marginBottom: 16, border: '1px solid #ffccc7' }}>
                    {detailError}
                  </div>
                )}
                {detailLoading && <p className="loading">加载链路数据中...</p>}
                {!detailLoading && !detailError && traceRuns.length === 0 && (
                  <p className="empty">暂无链路数据</p>
                )}
                {!detailLoading && !detailError && traceRuns.length > 0 && buildTraceTree(traceRuns).map(rootRun => (
                    <div key={rootRun.runId} style={{ marginBottom: 24 }}>
                      {/* 顶层 Agent */}
                      <div style={{
                        border: '1px solid #d9d9d9', borderRadius: 8, padding: '12px 16px',
                        background: '#fafafa', marginBottom: 8
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                          <div>
                            <strong>{agentTypeLabels[rootRun.agentType] || rootRun.agentType || 'Agent'}</strong>
                            <span style={{ color: '#888', fontSize: 12, marginLeft: 12 }}>
                              Run ID: {rootRun.runId}
                            </span>
                          </div>
                          <span className="status-badge" style={{
                            backgroundColor: (statusColors[normalizeStatus(rootRun.status)] || '#d9d9d9') + '20',
                            color: statusColors[normalizeStatus(rootRun.status)] || '#666',
                            fontSize: 12
                          }}>
                            {statusLabels[normalizeStatus(rootRun.status)] || rootRun.status}
                          </span>
                        </div>
                        {/* 用户输入 */}
                        {rootRun.input && (
                          <div style={{ color: '#555', fontSize: 13, marginBottom: 8, padding: '6px 10px', background: '#fff', borderRadius: 4, border: '1px solid #eee' }}>
                            {rootRun.input}
                          </div>
                        )}
                        {/* Steps */}
                        {rootRun.steps && rootRun.steps.length > 0 && (
                          <table className="admin-table small" style={{ marginBottom: 0 }}>
                            <thead><tr><th>#</th><th>步骤</th><th>类型</th><th>状态</th></tr></thead>
                            <tbody>
                              {rootRun.steps.map((s, i) => {
                                const ss = normalizeStatus(s.status);
                                return (
                                  <tr key={i}>
                                    <td style={{ width: 30, color: '#999' }}>{i + 1}</td>
                                    <td>{s.stepName}</td>
                                    <td>{stepTypeLabels[s.stepType] || s.stepType}</td>
                                    <td>
                                      <span className="status-badge" style={{
                                        backgroundColor: (statusColors[ss] || '#d9d9d9') + '20',
                                        color: statusColors[ss] || '#666', fontSize: 11
                                      }}>{statusLabels[ss] || s.status}</span>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        )}
                      </div>

                      {/* 子 Agent */}
                      {rootRun.children && rootRun.children.length > 0 && (
                        <div style={{ marginLeft: 32 }}>
                          {rootRun.children.map(child => (
                            <div key={child.runId} style={{
                              border: '1px solid #e8e8e8', borderRadius: 6, padding: '10px 14px',
                              background: '#fff', marginBottom: 6
                            }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                                <div>
                                  <span style={{ fontSize: 13 }}>{agentTypeLabels[child.agentType] || child.agentType || 'Worker'}</span>
                                  <span style={{ color: '#888', fontSize: 11, marginLeft: 8 }}>
                                    Run ID: {child.runId}
                                  </span>
                                </div>
                                <span className="status-badge" style={{
                                  backgroundColor: (statusColors[normalizeStatus(child.status)] || '#d9d9d9') + '20',
                                  color: statusColors[normalizeStatus(child.status)] || '#666', fontSize: 11
                                }}>
                                  {statusLabels[normalizeStatus(child.status)] || child.status}
                                </span>
                              </div>
                              {/* Worker 子问题 */}
                              {child.input && (
                                <div style={{ color: '#555', fontSize: 12, marginBottom: 6, padding: '4px 8px', background: '#f6f8fa', borderRadius: 4, border: '1px solid #e8e8e8' }}>
                                  {child.input}
                                </div>
                              )}
                              {child.steps && child.steps.length > 0 && (
                                <table className="admin-table small" style={{ marginBottom: 0 }}>
                                  <thead><tr><th>#</th><th>步骤</th><th>类型</th><th>状态</th></tr></thead>
                                  <tbody>
                                    {child.steps.map((s, i) => {
                                      const ss = normalizeStatus(s.status);
                                      return (
                                        <tr key={i}>
                                          <td style={{ width: 30, color: '#999' }}>{i + 1}</td>
                                          <td>{s.stepName}</td>
                                          <td>{stepTypeLabels[s.stepType] || s.stepType}</td>
                                          <td>
                                            <span className="status-badge" style={{
                                              backgroundColor: (statusColors[ss] || '#d9d9d9') + '20',
                                              color: statusColors[ss] || '#666', fontSize: 11
                                            }}>{statusLabels[ss] || s.status}</span>
                                          </td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                }
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
