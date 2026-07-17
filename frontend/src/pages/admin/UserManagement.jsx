import { useState, useEffect } from 'react';
import { userManagementAPI } from '../../api/admin';
import './AdminDashboard.css';

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  const loadUsers = async (page, size) => {
    setLoading(true);
    try {
      const p = page || pageNum;
      const s = size || pageSize;
      const response = await userManagementAPI.listUsers(p, s);
      setUsers(response.data.records || []);
      setTotal(response.data.total || 0);
    } catch (err) {
      console.error('加载用户列表失败:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsers(1);
  }, []);

  const totalPages = Math.ceil(total / pageSize);
  const goToPage = (page) => { if (page < 1 || page > totalPages) return; setPageNum(page); loadUsers(page); };
  const changePageSize = (size) => { setPageSize(size); setPageNum(1); loadUsers(1, size); };

  const handleStatusChange = async (userId, currentStatus) => {
    const newStatus = currentStatus === 1 ? 0 : 1;
    const action = newStatus === 1 ? '解封' : '封禁';

    if (!confirm(`确定要${action}此用户吗？`)) return;

    try {
      await userManagementAPI.updateUserStatus(userId, newStatus);
      loadUsers();
      alert(`${action}成功`);
    } catch (err) {
      console.error('更新用户状态失败:', err);
      alert(`${action}失败`);
    }
  };

  const getStatusBadge = (status) => {
    if (status === 1) {
      return <span className="status-badge normal">正常</span>;
    }
    return <span className="status-badge banned">已封禁</span>;
  };

  return (
    <div>
      <div className="page-header">
        <h1>用户管理</h1>
        <p>管理系统中的所有用户</p>
      </div>

      <div className="admin-card">
        <div className="card-header">
          <h2>用户列表</h2>
          <span className="total-info">共 {total} 个用户</span>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
            加载中...
          </div>
        ) : users.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
            暂无用户数据
          </div>
        ) : (
          <>
            <table className="admin-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>用户名</th>
                  <th>手机号</th>
                  <th>状态</th>
                  <th>注册时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id}>
                    <td>{user.id}</td>
                    <td>{user.username || '-'}</td>
                    <td>{user.phone || '-'}</td>
                    <td>{getStatusBadge(user.status)}</td>
                    <td>
                      {user.createTime
                        ? new Date(user.createTime).toLocaleString('zh-CN')
                        : '-'}
                    </td>
                    <td>
                      <button
                        className="action-btn edit"
                        onClick={() => handleStatusChange(user.id, user.status)}
                      >
                        {user.status === 1 ? '封禁' : '解封'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

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
          </>
        )}
      </div>
    </div>
  );
}
