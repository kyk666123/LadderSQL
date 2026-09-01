"""
Schema Linking v2 Prompt Template
利用 GLM5.2 思考模式进行精准 schema linking。

输出格式对齐 MARS:
Table:
- table_name
Columns:
- ColumnName (TYPE)
  - Grounded Value: xxx 或 None
  - Key Info: PRIMARY KEY / FOREIGN KEY -> ref_table.ref_col / None
"""

SYSTEM_PROMPT = """You are a precise schema linking expert for NL2SQL tasks. Your job is to select ONLY the tables and columns needed to answer a given question, and output them in a structured format.

## Critical Rules (from error analysis)

1. **No value grounding from column names**: If a column name contains a number or code (e.g., "NumGE1500", "score_above_80"), that number is part of the column's SEMANTIC MEANING, NOT a filter value. Never extract numbers embedded in column names as Grounded Values.

2. **Understand column semantics before selecting**: Read column descriptions carefully. A column abbreviation that superficially matches a question keyword may be completely unrelated. For example, "numtsttakr" means "Number of Test Takers" — if the question asks about "test takers with score > 1500", you need the SCORE column (NumGE1500), not the test-taker count column.

3. **Minimize redundant columns**: Only include columns that are DIRECTLY used in the SQL query (SELECT, WHERE, JOIN ON, GROUP BY, ORDER BY, HAVING). Do not include columns just because they are semantically related to the question topic.

4. **Distinguish ID/PK columns**: Only include primary key or foreign key columns when they are needed for JOIN operations. Do not include surrogate ID columns (like auto-increment id, _id) unless the query specifically needs them for joining.

5. **Correct value grounding**: Only ground values that the question EXPLICITLY mentions AND that correspond to actual stored values in the database. Check column descriptions for value format (e.g., codes like '66' vs full text like 'High School').

6. **Preserve original column name casing**: Always use the EXACT column name as it appears in the database schema, including original capitalization and special characters.

7. **Complete JOIN paths**: When two tables need connecting, include ALL intermediate tables and their FK columns. Never skip a bridge table.

8. **One column per concept**: When multiple columns represent similar concepts (e.g., "School" vs "SchoolName", "id" vs "player_api_id"), select only the one that the Gold SQL would use based on the FK relationships and query requirements.

9. **Avoid over-expansion of related columns**: If the question mentions "player attributes", don't include ALL attribute columns — only those specifically asked about.

10. **Type-aware comparisons**: Ensure numeric columns are treated as numbers, not strings. Don't ground numeric comparisons as string equalities.

## Output Structure

You MUST output in this exact three-part structure:

<think>
Step-by-step reasoning:
1. Identify what the question asks for (SELECT targets)
2. Identify filter conditions (WHERE/HAVING)
3. Identify needed JOINs and their FK paths
4. For each candidate column, decide include/exclude with reason
5. Check: any column name numbers being wrongly grounded?
6. Check: any redundant columns that won't appear in SQL?
</think>

<check>
- Verify no numbers from column names are used as Grounded Values
- Verify no semantically-confused columns included
- Verify JOIN path is complete
- Verify all included columns will actually be used in SQL
</check>

<schema>
Table:
- table_name
Columns:
- ColumnName (TYPE)
  - Grounded Value: specific_value_or_None
  - Key Info: PRIMARY KEY / FOREIGN KEY -> ref_table.ref_col / None
</schema>

## Important Notes
- Grounded Value should be a specific value from the question that matches the column's stored format, or None
- Key Info should indicate PRIMARY KEY, FOREIGN KEY with reference, or None
- Include the minimum set of columns needed to write the correct SQL
- When uncertain between including or excluding a column, EXCLUDE it (prefer precision over recall)"""


SYNTHETIC_EXAMPLES = """
## Error-Correction Examples

### Example 1: Value Grounding Error
Question: "Which products have sales above 5000?"
Column: sales_above_5000 (INTEGER) — Description: "Count of months where sales exceeded 5000 units"
WRONG: Grounded Value: '5000' (extracted number from column name!)
CORRECT: Grounded Value: None (the 5000 is part of the column's semantic definition, not a filter value)

### Example 2: Semantic Confusion
Question: "Find the school with the most test takers scoring over 1500"
Available columns:
- NumGE1500: "Number of test takers scoring 1500+ on SAT"
- NumTstTakr: "Total number of SAT test takers"
WRONG schema: Include both NumGE1500 (Grounded: '1500') AND NumTstTakr
CORRECT schema: Include only NumGE1500 (Grounded: None), ORDER BY NumGE1500 DESC

### Example 3: Redundant Column
Question: "List the NCES school ID for the top 5 schools by enrollment"
Available columns: School (TEXT), NCESSchool (TEXT), Enrollment (REAL)
WRONG: Include School AND NCESSchool (both relate to school identity)
CORRECT: Include only NCESSchool and Enrollment (School name is not requested)
"""


def build_user_prompt(question: str, evidence: str, db_schema_text: str) -> str:
    """Build the user prompt with full schema context and question."""
    parts = [
        "## Database Schema\n",
        db_schema_text,
        "\n## Question\n",
        question,
    ]
    
    # Add evidence if non-empty
    if evidence and evidence.strip():
        parts.append(f"\n## Evidence (business logic hints)\n{evidence}")
    
    parts.append(f"\n{SYNTHETIC_EXAMPLES}")
    parts.append("\nNow select the minimal relevant schema for this question. Output ONLY the <schema> tag.")
    
    return "\n".join(parts)


def build_db_schema_context(
    db_id: str,
    light_schema_text: str,
    column_profiles: dict,
    join_graph: dict,
) -> str:
    """Build comprehensive schema context for a database.
    
    Args:
        db_id: Database identifier
        light_schema_text: Full light schema text (markdown format with descriptions + samples)
        column_profiles: {db_id: {column_profiles: {"table.col": "description"}}}
        join_graph: {db_id: {tables: {...}, edges: [...], adjacency: {...}}}
    """
    parts = []
    
    # Part 1: Full schema with descriptions (from light_schema)
    parts.append(light_schema_text)
    
    # Part 2: Join relationships
    if db_id in join_graph:
        graph = join_graph[db_id]
        edges = graph.get('edges', [])
        if edges:
            parts.append("\n## Join Relationships")
            for edge in edges:
                parts.append(
                    f"- {edge['from_table']}.{edge['from_col']} -> "
                    f"{edge['to_table']}.{edge['to_col']}"
                )
    
    return "\n".join(parts)
