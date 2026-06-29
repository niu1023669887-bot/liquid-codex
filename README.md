# Liquid Codex

调酒配方平台，416种材料数据库。

## 这是什么

一个让AI生成的调酒配方至少在物理上可行的系统。不是那种随便让大模型瞎编配方的玩具。

核心功能：
- 416种材料数据库（pH值、糖度、密度等客观数据）
- 20条物理校验规则（稀释度、酸碱平衡、乳化安全等）
- AI配方生成 + 物理验证

## 本地运行

```bash
git clone https://github.com/niu1023669887-bot/liquid-codex.git
cd liquid-codex

pip install -r requirements.txt

cd backend
uvicorn main:app --reload --port 8000

# 另开终端
cd frontend
python -m http.server 3000
```

打开 http://localhost:3000

## 环境变量

在 `backend/.env` 里配置：

```
JWT_SECRET=随便写个长字符串
PASSWORD=你的管理员密码
DEEPSEEK_API_KEY=可选，用于AI功能
PERPLEXITY_API_KEY=可选，用于成分验证
```

## 部署

后端：Railway  
前端：Vercel

```bash
railway up
vercel deploy
```

## 致谢

- FastAPI - 后端框架
- DeepSeek - AI推理
- Perplexity - 成分数据验证
- Railway & Vercel - 部署平台
- Dave Arnold《Liquid Intelligence》- 稀释热力学理论
- Jim Meehan《Meehan's Bartender Manual》- 空间配比学

## 协议

MIT License
