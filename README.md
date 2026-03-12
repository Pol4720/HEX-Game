# HEX Game

<div align="center">

**AI-Powered HEX Board Game with Enhanced MCTS**

An implementation of the classic HEX board game featuring an intelligent AI agent based on Monte Carlo Tree Search with six weighted heuristics and Dijkstra-based win detection.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

</div>

---

## Overview

HEX is a two-player strategy board game played on an NxN rhombus grid where players take turns placing stones, aiming to connect their two opposite sides of the board. This project implements:

- A fully functional HEX game engine
- An AI player using an **enhanced Monte Carlo Tree Search (MCTS)** algorithm
- Multiple strategic heuristics that guide the search toward stronger moves

## AI Strategy

### Enhanced MCTS

The AI uses UCB1 (Upper Confidence Bound) for balancing exploration and exploitation during tree search, combined with six domain-specific heuristics:

| Heuristic | Purpose |
|-----------|---------|
| **Bridge Detection** | Identifies and prioritizes bridge patterns that create virtual connections |
| **Center Control** | Favors moves near the board center for strategic flexibility |
| **Shortest Path** | Uses Dijkstra's algorithm to evaluate progress toward connecting sides |
| **Connection Strength** | Assesses how well a move strengthens existing stone groups |
| **Blocking** | Prioritizes moves that disrupt the opponent's shortest connection path |
| **Edge Proximity** | Weighs moves near the player's target edges for endgame play |

### Win Detection

The game uses **Dijkstra's algorithm** to determine if a player has connected their two sides of the board, treating the board as a weighted graph where occupied cells have zero cost.

## Features

- Configurable board size (NxN)
- Human vs AI and AI vs AI modes
- Adjustable MCTS simulation count and time limit
- Tunable heuristic weights for strategy customization
- Visual board display in terminal

## Getting Started

```bash
git clone https://github.com/Pol4720/HEX-Game.git
cd HEX-Game

python main.py
```

## How HEX Works

```
  ╲ 1   2   3   4   5  ╲
 1 ╲  .   .   .   .   . ╲
  2 ╲  .   B   .   .   . ╲
   3 ╲  .   .   R   .   . ╲
    4 ╲  .   .   .   .   . ╲
     5 ╲  .   .   .   .   . ╲
```

- **Blue (B)** connects left ↔ right
- **Red (R)** connects top ↔ bottom
- No draws are possible — HEX always has a winner (Brouwer's fixed-point theorem)

## Academic Context

Developed as an **Artificial Intelligence** course project at the University of Havana, Faculty of Mathematics and Computer Science (MATCOM).

## License

This project is licensed under the MIT License.
