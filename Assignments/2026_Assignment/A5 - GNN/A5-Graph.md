# A5: Graph Neural Networks — Teaching Guide

---

## Opening: Why Graphs?

**What to say:**

> "Every model we've built assumes data lives on a grid — pixels in a 2D image, tokens in a sequence. But most real-world data doesn't look like that. Social networks, molecules, recommendation systems, knowledge graphs — they're all relational. GNNs are the tool for this."

**Pause and ask:**

> "You want to predict whether a user will like a movie. You have the user's rating history and the movie's genre. What extra information would help — and how would you represent it?"
>
> Answer: Other users who rated the same movies (collaborative filtering), similar movies (co-rating graph). The relationship structure is the information.

---

## Part 1: Graphs and Karate Club (Cells 3–4)

**What to say:**

> "Zachary's Karate Club is the MNIST of GNNs. 34 members of a karate club. A conflict happened — the club split into two factions. The graph shows who socialized with whom. Can a GNN predict which faction each member joined, using only the social links?"

**Key concepts to establish:**

- **Node** = entity (person, movie, atom)
- **Edge** = relationship (friends, co-rated, bonded)
- **Node features** = what we know about each entity
- **Labels** = what we want to predict

**The adjacency matrix A:**

> "A graph with N nodes = an N×N matrix where A[i][j]=1 if there's an edge. Most real graphs are sparse — most entries are 0. We represent this efficiently as an edge list, not a full matrix."

---

## Part 2: The MovieLens Graph (Cells 5–6)

**What to say:**

> "Instead of a toy graph, we'll work with MovieLens-100k — 1,682 movies, connected if the same user rated both. Each node has features: genre one-hot vector + release year. We'll predict each movie's primary genre."

**Why this is interesting:**

> "The graph structure encodes taste similarity — movies that get co-rated tend to be similar. A GNN can use this structure directly. A regular MLP can only use the node features (genre, year) — it ignores who watched what together."

---

## Part 3: GCN (Cells 7–11)

### The core idea — message passing

**What to say:**

> "The idea behind all GNNs is embarrassingly simple: to update a node's representation, aggregate features from its neighbors, then apply a learned transformation."

$$h_v^{(k)} = \sigma\left( W \cdot \text{AGGREGATE}\left(\{ h_u^{(k-1)} : u \in \mathcal{N}(v) \}\right) \right)$$

**Intuition for k layers:**

> "With 1 GCN layer, each node sees its immediate neighbors. With 2 layers, it sees neighbors-of-neighbors — 2-hop neighborhood. With k layers, k-hop. More layers = more context, but too many causes over-smoothing."

### GCN's normalization trick

$$H^{(k+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)} \right)$$

**What to say:**

> "Without normalization, nodes with many neighbors dominate — their features get summed 50 times while isolated nodes sum once. $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ divides by the square root of degree on both sides. Every neighbor contributes proportionally."

**The self-loop trick ($\tilde{A} = A + I$):**

> "Add the identity to the adjacency — every node becomes its own neighbor. Without this, a node's current representation is completely replaced by its neighbors' features. The self-loop preserves the node's own information."

---

## Part 4: GAT (Cells 12–16)

### The GCN limitation

**What to say:**

> "GCN treats all neighbors equally. But if you're a drama film, a co-rating with another drama tells you more than a co-rating with an action film. Some neighbors matter more."

**GAT's fix — learned attention:**

$$\alpha_{ij} = \text{softmax}_j\left( \text{LeakyReLU}\left( \mathbf{a}^T [W h_i \| W h_j] \right) \right)$$

> "For each edge (i→j), compute a scalar attention score from the concatenated features of both nodes. Softmax over all neighbors so weights sum to 1. Now aggregation is a *weighted* sum — the model learns which neighbors to trust."

**Multi-head attention:**

> "Run K independent attention mechanisms in parallel, concatenate their outputs. Same idea as Transformer multi-head attention. Different heads can learn to focus on different types of relationships."

**When does GAT beat GCN?**

> "When neighbor importance is heterogeneous — some neighbors are much more informative than others. On homogeneous graphs (all neighbors equally relevant), GAT and GCN perform similarly. GAT is also more robust to noisy edges."

### GCN vs GAT — connect to Transformers from A1

> "Attention in GAT is the same mechanism as in the Vision Transformer. ViT asks: which *patches* should attend to each other? GAT asks: which *neighbors* should a node attend to? Both learn to weight inputs dynamically instead of using fixed weights."

---

## Part 5: Recommendation System (Cells 17–19)

### The bipartite graph idea

**What to say:**

> "So far we predicted node labels (genre classification). GNNs can also predict *edges* — link prediction. In recommendation: does user U like movie M? That's predicting whether an edge should exist in the user-movie bipartite graph."

```
Users        Movies
  U1 ────── M1 (rated)
  U1 ────── M3 (rated)
  U2 ────── M1 (rated)
  U2 ──?──  M3 (not yet rated — should we recommend?)
```

**The GNN approach:**

> "Embed users and movies into the same vector space using GCN on the bipartite graph. Users who rated similar movies get similar embeddings. Movies rated by similar users get similar embeddings. Predict rating = dot product of user and movie embeddings."

**Why GNN beats matrix factorization:**

> "Matrix factorization only uses the rating matrix. GNN can additionally use node features — movie genre, user demographics. And the graph structure itself propagates information: if U1 and U2 have similar taste, M3's embedding is influenced by U1's ratings even though U1 never rated M3."

---

## Exercise Walkthrough Tips

**Ex1 (Over-smoothing):** With 1 layer, nodes only see 1-hop neighbors. With 5+ layers, all nodes start averaging over the whole graph — embeddings converge to the same value. Students should see accuracy *drop* after 3-4 layers. Plot accuracy vs depth.

**Ex2 (GCN vs GAT comparison):** Key metric is val accuracy on genre prediction. GAT should win on MovieLens because genre similarity is heterogeneous — drama films co-rated with drama tell you more than random co-ratings.

**Ex3 (Recommendation):** Link prediction metric is AUC or Hits@K (did the true movie appear in the top-K recommendations?). Students should compare against a non-graph baseline (popularity-based: recommend most-rated movies to everyone).

---

## The Big Picture

```
2009  GNN (Scarselli)    "First framework: aggregate neighbor messages iteratively."
        ↓
2017  GCN (Kipf)         "Spectral convolution → simple matrix form. Symmetric normalization."
        ↓
2018  GAT (Veličković)   "Learned attention weights per edge. Not all neighbors equal."
        ↓
2019  GraphSAGE          "Sample neighbors, not full graph. Scales to millions of nodes."
        ↓
2020+ Graph Transformers "Full attention over all nodes. No fixed graph structure needed."
```

**Closing question:**

> "You're building a drug discovery system. Atoms are nodes, chemical bonds are edges, node features are atom type and charge. You want to predict if a molecule is toxic. How many GCN layers would you use, and why?"
>
> Answer: 3–4 layers. A molecule's toxicity depends on functional groups (local structure, 2–3 hops) and global shape (4–5 hops). Too many layers → over-smoothing, all atoms look the same. Sweet spot: enough to capture the relevant subgraph structures.
