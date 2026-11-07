data = [(969,382,2670),(776,250,1504),(746,241,1485),(738,229,1372),(809,309,2049),(821,333,2275),(722,228,1410),(712,207,1147),(729,223,1278),(816,321,2200),(823,322,2181),(883,375,2651)]
import statistics
ts=[d[0] for d in data]; fs=[d[1] for d in data]; bs=[d[2] for d in data]
n=len(data)
mt=sum(ts)/n; mf=sum(fs)/n; mb=sum(bs)/n
cov = sum((a-mf)*(b-mb) for a,b in zip(fs,bs))/n
sdf = (sum((a-mf)**2 for a in fs)/n)**0.5
sdb = (sum((b-mb)**2 for b in bs)/n)**0.5
r = cov/(sdf*sdb)
print(f'RESULT corr(fetch_ms, disk_MB) = {r:.4f}')
# 线���拟合: fetch = a*disk + b
a = cov/(sdb*sdb); b = mf - a*mb
print(f'RESULT fetch_ms = {a*1000:.3f} us/MB * disk_MB + {b:.0f} ms')
print(f'RESULT => {a*1e3:.1f} us per MB = {1/(a*1e3)*1000:.2f} GB/s effective')
print()
print(f'RESULT mean fetch={mf:.0f}ms, mean disk={mb:.0f}MB')
print(f'RESULT 若命中率 100%（disk=0），fetch 预计 {b:.0f} ms (截距，即固定开销)')
print(f'RESULT 可优化空间: {mf-b:.0f} ms/token = {mf-b/mt*100:.0f}%')