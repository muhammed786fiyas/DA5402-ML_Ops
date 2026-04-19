# 🐳 DA5402 – Part 2: Docker & Docker Swarm Summary

---

# 1️⃣ Why Docker?

## 🔹 Problem Without Docker
- Different machines
- Different Python versions
- Different dependency versions
- “Works on my machine” issue

## 🔹 Solution
Docker packages:
- Code
- Python
- Dependencies
- Models
- Everything needed

Into one unit:

👉 **Container**

---

# 2️⃣ Core Docker Concepts

## 🧱 Image
- Blueprint/template
- Built using Dockerfile
- Example: `python:3.11-slim`

---

## 🐳 Container
- Running instance of image
- Isolated environment
- Runs your FastAPI app

---

## 📄 Dockerfile
Text file that defines how image is built.

Example structure:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🔓 Exposing Ports

Two ports exist:

- Container port (inside container)
- Host port (machine port)

Example:

```
-p 8000:8000
```

Means:

```
Machine Port 8000 → Container Port 8000
```

---

## ⚠️ EXPOSE vs -p

- `EXPOSE 8000` → documentation only
- `-p 8000:8000` → actually publishes port

---

# 3️⃣ Docker Swarm (Cluster Mode)

Swarm allows:
- Multiple machines
- Multiple containers
- Load balancing
- High availability

---

# 4️⃣ Nodes

A **node** = A machine in swarm cluster.

Types:

### 🔹 Manager Node
- Controls cluster
- Schedules services

### 🔹 Worker Node
- Runs containers

Assignment setup:

- Machine 1 → Manager  
- Machine 2 → Worker  

---

# 5️⃣ Service, Task, Container Hierarchy

Hierarchy:

Node → Service → Task → Container

---

## 🔹 Service
Definition of what to run.

Example:
```yaml
replicas: 4
```

---

## 🔹 Task
One replica created by service.

---

## 🔹 Container
Actual running instance of app.

---

If you deploy:

```yaml
replicas: 4
```

Swarm creates:
- 4 tasks
- 4 containers
- Distributed across nodes

---

# 6️⃣ Replicas

Replica = One copy of container.

If:
```yaml
replicas: 4
```

Then 4 containers run.

Swarm decides placement automatically.

Likely:
- 2 on Machine 1
- 2 on Machine 2

But scheduler decides based on resources.

---

# 7️⃣ Overlay Network

Overlay network:

👉 Virtual network connecting containers across machines.

Allows:
- Cross-node communication
- Service discovery
- Internal routing

Without overlay:
- Containers cannot communicate across machines easily.

---

# 8️⃣ Ingress Mesh

Ingress mesh = Built-in load balancer.

Behavior:

- Port 8000 exposed on all nodes
- You can send request to any node
- Swarm forwards request to any replica

You do NOT choose container.
Swarm load balancer chooses.

---

# 9️⃣ Why container_id Was Important

You added:

```python
socket.gethostname()
```

This returns container ID.

When 100 requests are sent:

```
Counter({'A1': 25, 'A2': 24, 'B1': 26, 'B2': 25})
```

This proves:
- Load balancing works
- Traffic distributed across replicas

---

# 🔟 Final Architecture for Assignment

You will have:

- 2 Nodes
- 1 Service
- 4 Replicas
- Overlay Network
- Ingress Mesh

Flow:

Requests → Swarm → Distributed Containers → Response with container_id

---

# 🎯 Viva Preparation

Be able to explain:

- What is Docker?
- What is an image?
- What is a container?
- What is a service?
- What is a task?
- What is a node?
- What is overlay network?
- What is ingress mesh?
- How load balancing happens?

---

# 🏁 Big Picture

Part 1 = Build API  
Part 2 = Deploy API at scale  

You move from:

AI Developer  
→  
Distributed Systems Engineer