# MTP 的收益账
main_experts_per_token = 40 * 6      # 240
EXP = 18.8  # MB
# 不开 MTP：N 个 token，每个 240 次专家 H2D
def cost_no_mtp(n_tok):
    experts = n_tok * main_experts_per_token
    return experts, experts * EXP / 1024
# 开 MTP：每轮 draft 5 个，接受 a 个
# 主 forward 一次验证 5 个 -> 240 次专家 H2D（一次 forward 内去重后可能少些）
# draft: 3 层 x 5 token x 6 = 90 次
def cost_mtp(n_tok, accept_rate):
    rounds = n_tok / (1 + 5*accept_rate)
    main = rounds * main_experts_per_token
    draft = rounds * 3 * 6 * 5
    tot = main + draft
    return tot, tot * EXP / 1024
print('每生成 10 个 token：')
e0, g0 = cost_no_mtp(10)
print(f'  不开 MTP: {e0:.0f} 次专家传输 = {g0:.1f} GB')
for a in (0.3, 0.5, 0.7, 0.9):
    e, g = cost_mtp(10, a)
    print(f'  MTP 接受率 {a*100:.0f}%: {e:.0f} 次 = {g:.1f} GB  ({(1-e/e0)*100:+.0f}%)')
print()
print('注意：MTP 的收益来自「一次 forward 验证多个 token」，但专家传输是按 forward 算的，')
print('所以真正的节省在于：主 forward 的 240 次专家访问覆盖了 1+5a 个 token。')
print()
print('关键：draft 层也要 7.6GB，8GB 卡装不下。但 draft 的专家可以走同一条流式路径。')