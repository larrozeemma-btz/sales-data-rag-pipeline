# Retrieval evaluation

60 labelled questions (10 per type), precision@5: the share of the top 5 rows that satisfy every condition in the question.

| Question type | TF-IDF only | Constraint filter + TF-IDF |
|---|---|---|
| Product + region (German name) | 100% | 100% |
| Category + customer type | 100% | 100% |
| Product + region (German name) + discount | 8% | 100% |
| Product + region (English name) + discount | 2% | 100% |
| Category + region (English name) + customer type | 2% | 100% |
| Product nickname + region (English name) + discount | 0% | 32% |
| **All questions** | **35%** | **89%** |
