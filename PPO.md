# PPO 与 GAE 原理简介

## 🧠 Proximal Policy Optimization (PPO)

PPO（近端策略优化）是一种强化学习中常用的策略梯度方法，旨在在保持学习稳定性的同时提高样本效率。PPO 的核心思想是通过限制每次策略更新的幅度，避免策略发生剧烈变化，从而实现稳定训练。

### PPO 的主要特点：

1. **剪切目标函数（Clipped Objective）**：
   PPO 不直接最大化期望回报，而是通过引入剪切机制来限制策略更新幅度：
   
   $$L^{CLIP}(\theta) = \mathbb{E}\_t \left[ \min\left( r\_t(\theta)\hat{A}\_t, \text{clip}(r\_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}\_t \right) \right]$$
   其中：
   - $r\_t(\theta) = \frac{\pi\_\theta(a\_t | s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t | s\_t)}$ 是新旧策略的概率比；
   - $\hat{A}\_t$ 是优势函数（可由 GAE 估算）；
   - $\epsilon$ 是一个超参数（如 0.1 或 0.2）。

2. **信赖区域的近似优化**：
   PPO 使用剪切来代替复杂的二阶优化方法（如 Trust Region Policy Optimization, TRPO），降低了实现和计算的复杂度，用于防止新旧策略差异过大。

3. **值函数防止过度优化**：
    PPO为了防止过度优化，保证新策略和源策略不会差异太大，在奖励函数中添加了一个KL惩罚：
    $$r\_t = t\_\phi(q, o\_{\leq t}) - \beta \text{log}\frac{\pi\_{\theta}(o\_t|q,o\_{<t})}{\pi\_{ref}(o\_t|q,o\_{<t})}$$


**具体的优化原理：**

当 $\hat{A}\_t$ 为正时，表示当前行为 $a\_t$ 相对于旧策略是“有利”的，我们希望提高该行为的概率，因此期望优化参数 $\theta$ 使得 $r\_t(\theta) = \frac{\pi\_\theta(a\_t | s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t | s\_t)}$ 越大越好。但由于引入了剪切（clip）操作，为了防止策略更新过大，$r\_t(\theta)$ 的值被限制在 $[1 - \epsilon, 1 + \epsilon]$ 之间，因此实际目标最多只会增加到 $(1 + \epsilon)\hat{A}\_t$。

相反，当 $\hat{A}\_t$ 为负时，表明该行为是不利的，我们希望减小其概率，即期望优化参数 $\theta$ 使得$r\_t(\theta) = \frac{\pi\_\theta(a\_t | s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t | s\_t)}$ 越小越好，但由于有clip的限制，因此其最小不会超过 $(1 - \epsilon)\hat{A}\_t$ 。

---

## ⚖️ Generalized Advantage Estimation (GAE)

### 🧠 为什么需要 GAE？

在策略梯度算法中，我们需要估算一个值叫做**优势函数（Advantage Function）**，它表示：

> 某个动作比平均水平好多少？

优势函数决定了我们是否应该“鼓励”这个动作。

但问题是，这个值我们得**自己估算**，而如何估得又**准确（偏差小）**、又**稳定（方差小）**，就成了关键。

---

### 💡 最简单的两个估算方法

#### ⏱️ 方法一：MC（Monte Carlo）/ TD(1)
用完整的未来收益预估

$$Adv = R\_t + γ R\_{t+1} + γ^2 R\_{t+2} + ... - V(s\_t)$$
- ✅ 优点：平均来说更准确（无偏）
- ❌ 缺点：方差很大（因为未来太多不确定了）

#### ⏱️ 方法二：TD(0)
只看一步

$$Adv = r\_t + γ * V(s\_{t+1}) - V(s\_t)$$
- ✅ 优点：很稳定（方差小）
- ❌ 缺点：不准确（有偏）

其中 $r\_t$ 表示第$t$步时从环境中得到的及时奖励，由Reward Model给出，表示为：

$$r\_t = \text{Reward}(s\_t, a\_t)$$

而 $V(s\_t)$ 是状态 $s\_t$ 的价值，表示在状态 $s\_t$ 下，期望获得的总回报（reward）。状态价值的计算通常依赖于价值函数估计方法，如 蒙特卡罗方法、时序差分方法（TD），或者通过神经网络近似。

### ⚖️ GAE的核心思想：在中间找一个平衡

GAE 把多个时间步的 TD 残差（TD Error）按一定权重加起来，既不只看一步，也不看太远，而是逐步衰减：

#### 🔧 数学表达：
定义每一步的 TD 残差为：

$$\delta\_t = r\_t + \gamma V(s\_{t+1}) - V(s\_t)$$

这是当前时刻的即时奖励加上折扣后的下一个状态的估计价值，减去当前状态的价值。

GAE 把多个 TD 残差按 $(\gamma \lambda)^l$ 加权求和：

$$\hat{A}\_t^{\text{GAE}(\gamma, \lambda)} = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}$$
 
其中：

$\gamma$ 是折扣因子（如 0.99）

$\lambda$ 是 GAE 的平衡参数（如 0.95）

$\lambda = 0$ → 只使用一步 TD（稳定但偏差大）

$\lambda = 1$ → 接近完整的 Monte Carlo（无偏但方差大）

GAE 允许我们在这两者之间找到一个最优平衡点。

### 👶 举个通俗例子：
你是一个学生，要估计“这门课这学期值不值得继续努力”。

TD(0)：你只看这周成绩 + 预测下周 → 太短视。

MC：你等期末看总分 → 太迟了，而且容易被某次波动影响。

GAE：你综合考虑未来几周的表现，但越远的成绩你看的越少 → 稳中求准

