import { useState, useEffect } from 'react';
import './AdminDashboard.css';

const phaseColors = {
  HTTP: '#722ed1', DB: '#1890ff', CACHE: '#13c2c2', CONTEXT: '#2f54eb',
  AI_CALL: '#fa8c16', INTENT: '#52c41a', ROUTE: '#eb2f96', REWRITE: '#faad14',
  RETRIEVAL: '#1677ff', RERANK: '#fa8c16', GENERATION: '#f5222d', TOOL: '#722ed1',
  REACT: '#eb2f96', SAFETY: '#ff4d4f', RESPONSE: '#52c41a',
};

const phaseLabels = {
  HTTP: 'HTTP层', DB: '数据库', CACHE: '缓存', CONTEXT: '上下文',
  AI_CALL: 'AI调用', INTENT: '意图识别', ROUTE: '路由分发', REWRITE: '问题改写',
  RETRIEVAL: '检索', RERANK: '重排序', GENERATION: 'LLM生成', TOOL: '工具调用',
  REACT: 'ReAct循环', SAFETY: '安全审查', SESSION: '会话',
  CLARIFY: '澄清追问', GUARD: '健康守卫', RESPONSE: '响应',
};

const sourceLabels = { java: 'Java', python: 'Python' };
const sourceColors = { java: '#1677ff', python: '#52c41a' };

export default function RequestTraceManagement() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detailEvents, setDetailEvents] = useState([]);
  const [detailMeta, setDetailMeta] = useState({});
  const [detailTraceId, setDetailTraceId] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState(null);

  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  const [filters, setFilters] = useState({ userType: '', userId: '', status: '', dateStart: '', dateEnd: '' });

  const fetchList = (page, size) => {
    setLoading(true);
    const p = page || pageNum;
    const s = size || pageSize;
    let url = `/api/v1/traces/list?pageNum=${p}&pageSize=${s}`;
    if (filters.userType) url += `&userType=${filters.userType}`;
    if (filters.userId) url += `&userId=${filters.userId}`;
    if (filters.status) url += `&status=${filters.status}`;
    if (filters.dateStart) url += `&startTime=${filters.dateStart}`;
    if (filters.dateEnd) url += `&endTime=${filters.dateEnd}`;

    const token = localStorage.getItem('adminToken');
    fetch(url, { headers: { 'Content-Type': 'application/json', 'Authorization': token ? `Bearer ${token}` : '' } })
      .then(r => r.json())
      .then(data => {
        setData(data.records || []);
        setTotal(data.total || 0);
      })
      .catch(e => console.error('获取Trace列表失败:', e))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchList(1); }, []);

  const handleSearch = () => { setPageNum(1); fetchList(1); };
  const handleReset = () => { setFilters({ userType: '', userId: '', status: '', dateStart: '', dateEnd: '' }); setPageNum(1); fetchList(1); };
  const goToPage = (page) => { if (page < 1 || page > totalPages) return; setPageNum(page); fetchList(page); };
  const changePageSize = (size) => { setPageSize(size); setPageNum(1); fetchList(1, size); };

  const openDetail = (traceId) => {
    setDetailTraceId(traceId);
    setDetailVisible(true);
    setDetailLoading(true);
    setDetailEvents([]);
    const token = localStorage.getItem('adminToken');
    fetch(`/api/v1/traces/${encodeURIComponent(traceId)}`, {
      headers: { 'Content-Type': 'application/json', 'Authorization': token ? `Bearer ${token}` : '' }
    })
      .then(r => r.json())
      .then(data => {
        if (data && data.events) {
          setDetailMeta({ durationMs: data.durationMs, status: data.status });
          setDetailEvents(data.events || []);
        } else if (Array.isArray(data)) {
          setDetailMeta({});
          setDetailEvents(data);
        } else {
          setDetailMeta({});
          setDetailEvents([]);
        }
      })
      .catch(e => console.error('获取Trace详情失败:', e))
      .finally(() => setDetailLoading(false));
  };

  const handleDeleteClick = (traceId) => {
    setDeleteTargetId(traceId);
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = () => {
    if (!deleteTargetId) return;
    const token = localStorage.getItem('adminToken');
    fetch(`/api/v1/traces/${encodeURIComponent(deleteTargetId)}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json', 'Authorization': token ? `Bearer ${token}` : '' }
    })
      .then(() => { setShowDeleteConfirm(false); setDeleteTargetId(null); fetchList(pageNum); })
      .catch(e => console.error('删除失败:', e));
  };

  const handleDeleteCancel = () => { setShowDeleteConfirm(false); setDeleteTargetId(null); };

  const formatTime = (time) => {
    if (!time) return '-';
    try {
      const d = new Date(time);
      return d.getFullYear() + '/' + (d.getMonth()+1) + '/' + d.getDate()
        + ' ' + String(d.getHours()).padStart(2,'0') + ':' + String(d.getMinutes()).padStart(2,'0')
        + ':' + String(d.getSeconds()).padStart(2,'0') + '.' + String(d.getMilliseconds()).padStart(3,'0');
    }
    catch { return String(time).substring(0, 23); }
  };

  const formatMs = (ms) => {
    if (ms == null) return '-';
    if (ms < 1000) return ms + 'ms';
    return (ms / 1000).toFixed(2) + 's';
  };

  const truncate = (s, max) => {
    if (!s) return '-';
    const str = typeof s === 'string' ? s : JSON.stringify(s);
    return str.length > max ? str.substring(0, max) + '...' : str;
  };

  const parseJson = (val) => {
    if (!val) return null;
    if (typeof val === 'object') return val;
    try { return JSON.parse(val); } catch { return val; }
  };

  const totalMs = detailMeta.durationMs
    || (detailEvents.length > 1
        ? new Date(detailEvents[detailEvents.length-1].eventTime) - new Date(detailEvents[0].eventTime)
        : 0);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="admin-panel">
      <div className="page-header">
        <h1>全链路请求追踪</h1>
        <p>查看用户请求的完整执行链路</p>
      </div>

      {/* 筛选 */}
      <div className="search-bar">
        <div className="search-row">
          <div className="search-item">
            <label>用户类型:</label>
            <select value={filters.userType} onChange={e => setFilters({...filters, userType: e.target.value})} >
              <option value="">全部</option>
              <option value="user">普通用户</option>
              <option value="admin">管理员</option>
            </select>
          </div>
          <div className="search-item">
            <label>用户ID:</label>
            <input value={filters.userId} onChange={e => setFilters({...filters, userId: e.target.value})} placeholder="输入用户ID" style={{ width: 120 }}/>
          </div>
          <div className="search-item">
            <label>状态:</label>
            <select value={filters.status} onChange={e => setFilters({...filters, status: e.target.value})}>
              <option value="">全部</option>
              <option value="SUCCESS">成功</option>
              <option value="FAILED">失败</option>
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
              <th>Trace ID</th>
              <th>用户</th>
              <th>事件数</th>
              <th>耗时</th>
              <th>状态</th>
              <th>时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan="7" className="loading">加载中...</td></tr>
            ) : data.length === 0 ? (
              <tr><td colSpan="7" className="empty">暂无数据（发起一次对话后即可看到 trace）</td></tr>
            ) : (
              data.map(item => (
                <tr key={item.traceId}>
                  <td title={item.traceId} style={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {item.traceId && item.traceId.length > 28
                      ? item.traceId.substring(0, 28) + '...'
                      : (item.traceId || '-')}
                  </td>
                  <td>{item.userId || '-'}</td>
                  <td>{item.eventCount}</td>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{formatMs(item.durationMs)}</td>
                  <td>
                    <span className="status-badge" style={{
                      background: (item.status === 'SUCCESS' ? '#52c41a' : '#ff4d4f') + '18',
                      color: item.status === 'SUCCESS' ? '#52c41a' : '#ff4d4f',
                      fontSize: 11, fontWeight: 500
                    }}>{item.status || 'SUCCESS'}</span>
                  </td>
                  <td style={{ fontSize: 12 }}>{formatTime(item.startTime)}</td>
                  <td>
                    <button className="action-btn edit" onClick={() => openDetail(item.traceId)}>查看</button>
                    <button className="action-btn delete" style={{ marginLeft: 8 }}
                      onClick={() => handleDeleteClick(item.traceId)}>删除</button>
                  </td>
                </tr>
              ))
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

      {/* ===== Trace 详情弹窗 ===== */}
      {detailVisible && (
        <div className="modal-overlay" onClick={() => setDetailVisible(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}
            style={{ maxWidth: 1100, width: '95%', maxHeight: '90vh', overflow: 'auto' }}>
            <div className="modal-header">
              <h3>Trace 链路 — {detailTraceId}</h3>
              <button className="modal-close" onClick={() => setDetailVisible(false)}>×</button>
            </div>
            <div className="modal-body">
              {detailLoading ? (
                <p className="loading">加载链路数据中...</p>
              ) : detailEvents.length === 0 ? (
                <p className="empty">暂无链路数据</p>
              ) : (
                <div>
                  <div style={{
                    marginBottom: 16, padding: '12px 16px', background: '#f9f9f9',
                    borderRadius: 8, display: 'flex', gap: 32, fontSize: 13, color: '#555', flexWrap: 'wrap'
                  }}>
                    <span>总事件数: <strong>{detailEvents.length}</strong></span>
                    <span>总耗时: <strong>{formatMs(totalMs)}</strong></span>
                    <span>状态: <strong style={{ color: detailMeta.status === 'FAILED' ? '#ff4d4f' : '#52c41a' }}>
                      {detailMeta.status || 'SUCCESS'}</strong></span>
                    <span>开始: <strong>{formatTime(detailEvents[0]?.eventTime)}</strong></span>
                    <span>结束: <strong>{formatTime(detailEvents[detailEvents.length - 1]?.eventTime)}</strong></span>
                  </div>

                  <div>
                    {detailEvents.map((event, idx) => {
                      const phase = event.phase || '';
                      const phaseColor = phaseColors[phase] || '#8c8c8c';
                      const inputObj = parseJson(event.inputData);
                      const outputObj = parseJson(event.outputData);
                      const hasTokens = event.inputTokens != null || event.outputTokens != null || event.totalTokens != null;
                      return (
                        <div key={idx} style={{ display: 'flex', gap: 12, padding: '8px 0', borderBottom: '1px solid #f0f0f0', alignItems: 'flex-start' }}>
                          <div style={{ width: 60, flexShrink: 0, textAlign: 'center' }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color: sourceColors[event.source] || '#666' }}>{event.stepOrder}</div>
                            <span style={{ fontSize: 10, display: 'inline-block', padding: '0 4px', borderRadius: 4, background: (sourceColors[event.source] || '#eee') + '18', color: sourceColors[event.source] || '#666', fontWeight: 600 }}>
                              {sourceLabels[event.source] || event.source}
                            </span>
                          </div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
                              <strong style={{ fontSize: 13 }}>{event.eventType}</strong>
                              {phase && <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: phaseColor + '18', color: phaseColor, fontWeight: 500 }}>{phaseLabels[phase] || phase}</span>}
                              {event.agentName && <span style={{ fontSize: 11, color: '#8c8c8c' }}>{event.agentName}</span>}
                              {event.modelName && <span style={{ fontSize: 11, color: '#bfbfbf' }}>{event.modelName}</span>}
                              {hasTokens && <span style={{ fontSize: 10, color: '#bfbfbf' }}>inputTokens:{event.inputTokens ?? '-'}  outputTokens:{event.outputTokens ?? '-'}  totalTokens:{event.totalTokens ?? '-'}</span>}
                              {event.latencyMs != null && <span style={{ fontSize: 11, fontWeight: 600, color: event.latencyMs > 1000 ? '#ff4d4f' : '#52c41a' }}>{formatMs(event.latencyMs)}</span>}
                            </div>
                            {inputObj && <div style={{ fontSize: 11, color: '#666', background: '#fafafa', padding: '4px 8px', borderRadius: 4, marginBottom: 4, maxHeight: 80, overflow: 'auto', whiteSpace: 'pre-wrap' }}><span style={{ color: '#999', fontWeight: 500 }}>INPUT </span>{truncate(inputObj, 300)}</div>}
                            {outputObj && (() => {
                              const { chunks, chunk_count, avg_score, doc_count, ...rest } = outputObj;
                              const isParentChild = chunks && chunks.length > 0 && chunks[0].parent_id != null;
                              const hasRest = Object.keys(rest).length > 0;
                              return (
                              <div style={{ fontSize: 11, color: '#333', background: '#f6ffed', padding: '4px 8px', borderRadius: 4, marginBottom: 4, maxHeight: 150, overflow: 'auto' }}>
                                <span style={{ color: '#52c41a', fontWeight: 500 }}>OUTPUT </span>
                                {hasRest && <span style={{ whiteSpace: 'pre-wrap' }}>{truncate(rest, 250)}</span>}
                                {chunks && chunks.length > 0 && (
                                  <>
                                    <table style={{ width: '100%', marginTop: 4, fontSize: 10, borderCollapse: 'collapse' }}>
                                      <thead>
                                        <tr style={{ background: '#e6f7e6' }}>
                                          <th style={{ padding: '2px 6px', border: '1px solid #d9d9d9' }}>#</th>
                                          <th style={{ padding: '2px 6px', border: '1px solid #d9d9d9' }}>doc_id</th>
                                          <th style={{ padding: '2px 6px', border: '1px solid #d9d9d9' }}>{isParentChild ? 'parent_id' : 'chunk_index'}</th>
                                          <th style={{ padding: '2px 6px', border: '1px solid #d9d9d9' }}>page</th>
                                          <th style={{ padding: '2px 6px', border: '1px solid #d9d9d9' }}>score</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {chunks.map((c, ci) => (
                                          <tr key={ci}>
                                            <td style={{ padding: '2px 6px', border: '1px solid #f0f0f0', textAlign: 'center' }}>{ci + 1}</td>
                                            <td style={{ padding: '2px 6px', border: '1px solid #f0f0f0', textAlign: 'center' }}>{c.doc_id ?? '-'}</td>
                                            <td style={{ padding: '2px 6px', border: '1px solid #f0f0f0', textAlign: 'center', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>{c.parent_id ?? c.chunk_index ?? '-'}</td>
                                            <td style={{ padding: '2px 6px', border: '1px solid #f0f0f0', textAlign: 'center' }}>{c.page ?? '-'}</td>
                                            <td style={{ padding: '2px 6px', border: '1px solid #f0f0f0', textAlign: 'center' }}>{c.score ?? '-'}</td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </>
                                )}
                              </div>
                            )})()}
                          </div>
                          <div style={{ width: 180, flexShrink: 0, fontSize: 11, color: '#999', textAlign: 'right', paddingTop: 2, whiteSpace: 'nowrap' }}>{formatTime(event.eventTime)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ===== 删除确认弹窗 ===== */}
      {showDeleteConfirm && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header"><h3>确认删除</h3></div>
            <div className="modal-body"><p>确定删除此条请求追踪记录吗？</p></div>
            <div className="modal-footer">
              <button className="modal-btn cancel" onClick={handleDeleteCancel}>取消</button>
              <button className="modal-btn confirm" onClick={handleDeleteConfirm}>确定</button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.45); z-index: 1000; display: flex; align-items: center; justify-content: center; }
        .modal-content { background: #fff; border-radius: 8px; box-shadow: 0 4px 24px rgba(0,0,0,0.15); display: flex; flex-direction: column; max-height: 90vh; }
        .modal-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid #f0f0f0; }
        .modal-header h3 { margin: 0; font-size: 16px; }
        .modal-close { background: none; border: none; font-size: 22px; color: #999; cursor: pointer; }
        .modal-body { padding: 16px 24px; overflow-y: auto; }
        .modal-footer { display: flex; justify-content: flex-end; gap: 12px; padding: 12px 24px; border-top: 1px solid #f0f0f0; }
        .modal-btn { padding: 6px 20px; border-radius: 4px; border: 1px solid #d9d9d9; cursor: pointer; font-size: 13px; }
        .modal-btn.confirm { background: #ff4d4f; color: #fff; border-color: #ff4d4f; }
        .modal-btn.cancel { background: #fff; color: #333; }
      `}</style>
    </div>
  );
}
