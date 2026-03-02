import logging
from typing import Dict, List, Optional

from sqlglot import parse_one, exp
from sqlglot.optimizer.qualify import qualify

from .execution import execute_sql
from .db_info import get_table_all_columns, get_db_all_tables

def is_sql_do_math(sql: str, include_count=False) -> bool:
    """
    Checks if the SQL query performs math operations such as AVG/SUM or +, -, *, /.
    
    Args:
        sql (str): The SQL query string.
    """
    try:
        parsed_sql = parse_one(sql, read='sqlite')
        math_aggs = {"AVG", "SUM", "MIN", "MAX"}
        if include_count: math_aggs.add("COUNT")
        math_expressions = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)

        first_select = next(parsed_sql.find_all(exp.Select), None)
        if not first_select:
            return False

        for select_exp in first_select.expressions:
            expression_body = select_exp.this if isinstance(select_exp, exp.Alias) else select_exp

            for math_exp in math_expressions:
                if next(expression_body.find_all(math_exp), None):
                    return True

            for agg in expression_body.find_all(exp.AggFunc):
                if agg.key and agg.key.upper() in math_aggs:
                    return True

        return False
    except Exception as e:
        logging.critical(f"Error in is_sql_do_math: {e}\nSQL: {sql}")
        raise e

def is_sql_to_cast(sql: str, include_subqueries: bool = False) -> bool:
    """
    Checks whether the SQL query uses CAST(...) in the SELECT projection list.

    Args:
        sql (str): The SQL query string.
        include_subqueries (bool): If True, checks all SELECTs including subqueries/CTEs.
            If False, only checks the outermost SELECT.
    """
    try:
        parsed_sql = parse_one(sql, read="sqlite")
        selects = list(parsed_sql.find_all(exp.Select))
        if not selects:
            return False

        def _select_has_cast(sel: exp.Select) -> bool:
            for select_exp in sel.expressions:
                expression_body = select_exp.this if isinstance(select_exp, exp.Alias) else select_exp
                if isinstance(expression_body, exp.Cast):
                    return True
                if next(expression_body.find_all(exp.Cast), None) is not None:
                    return True
            return False

        if include_subqueries:
            return any(_select_has_cast(sel) for sel in selects)
        return _select_has_cast(selects[0])
    except Exception as e:
        logging.critical(f"Error in is_sql_to_cast: {e}\nSQL: {sql}")
        raise e

def is_sql_use_join(sql: str, include_subqueries: bool = False) -> bool:
    """
    Checks whether the SQL uses JOINs in the FROM clause.

    Note:
        sqlglot represents both explicit JOIN syntax (e.g., "FROM a JOIN b ...") and
        comma-separated tables (e.g., "FROM a, b") via Join nodes, so this function
        treats both forms as "uses join".

    Args:
        sql (str): The SQL query string.
        include_subqueries (bool): If True, checks JOIN usage anywhere in the query (including subqueries/CTEs).
            If False, only checks JOIN usage in top-level SELECT statements (excluding subqueries/CTEs).
    """
    try:
        parsed_sql = parse_one(sql, read="sqlite")

        if include_subqueries:
            return next(parsed_sql.find_all(exp.Join), None) is not None

        # "Top-level" here means: not under a Subquery and not inside a CTE definition.
        def _is_top_level_select(sel: exp.Select) -> bool:
            return sel.find_ancestor(exp.Subquery) is None and sel.find_ancestor(exp.CTE) is None

        for sel in parsed_sql.find_all(exp.Select):
            if not _is_top_level_select(sel):
                continue
            if sel.args.get("joins"):
                return True
        return False
    except Exception as e:
        logging.critical(f"Error in is_sql_use_join: {e}\nSQL: {sql}")
        raise e

def get_sql_tables(db_path: str, sql: str) -> List[str]:
    """
    Retrieves table names involved in an SQL query.
    
    Args:
        db_path (str): Path to the database file.
        sql (str): The SQL query string.
        
    Returns:
        List[str]: List of table names involved in the SQL query.
    """
    db_tables = get_db_all_tables(db_path)
    try:
        parsed_tables = list(parse_one(sql, read='sqlite').find_all(exp.Table))
        correct_tables = [
            str(table.name).strip().replace('\"', '').replace('`', '') 
            for table in parsed_tables
            if str(table.name).strip().lower() in [db_table.lower() for db_table in db_tables]
        ]
        return correct_tables
    except Exception as e:
        logging.critical(f"Error in get_sql_tables: {e}\nSQL: {sql}")
        raise e

def _get_main_parent(expression: exp.Expression) -> Optional[exp.Expression]:
    """
    Retrieves the main parent expression for a given SQL expression.
    
    Args:
        expression (exp.Expression): The SQL expression.
        
    Returns:
        Optional[exp.Expression]: The main parent expression or None if not found.
    """
    parent = expression.parent
    while parent and not isinstance(parent, exp.Subquery):
        parent = parent.parent
    return parent

def _get_table_with_alias(parsed_sql: exp.Expression, alias: str) -> Optional[exp.Table]:
    """
    Retrieves the table associated with a given alias.
    
    Args:
        parsed_sql (exp.Expression): The parsed SQL expression.
        alias (str): The table alias.
        
    Returns:
        Optional[exp.Table]: The table associated with the alias or None if not found.
    """
    return next((table for table in parsed_sql.find_all(exp.Table) if table.alias == alias), None)

def get_sql_columns_dict(db_path: str, sql: str) -> Dict[str, List[str]]:
    """
    Retrieves a dictionary of tables and their respective columns involved in an SQL query.
    
    Args:
        db_path (str): Path to the database file.
        sql (str): The SQL query string.
        
    Returns:
        Dict[str, List[str]]: Dictionary of tables and their columns.
    """
    sql = qualify(parse_one(sql, read='sqlite'), qualify_columns=True, validate_qualify_columns=False) if isinstance(sql, str) else sql
    columns_dict = {}

    sub_queries = [subq for subq in sql.find_all(exp.Subquery) if subq != sql]
    for sub_query in sub_queries:
        subq_columns_dict = get_sql_columns_dict(db_path, sub_query)
        for table, columns in subq_columns_dict.items():
            if table not in columns_dict:
                columns_dict[table] = columns
            else:
                columns_dict[table].extend([col for col in columns if col.lower() not in [c.lower() for c in columns_dict[table]]])

    for column in sql.find_all(exp.Column):
        column_name = column.name
        table_alias = column.table
        table = _get_table_with_alias(sql, table_alias) if table_alias else None
        table_name = table.name if table else None

        if not table_name:
            candidate_tables = [t for t in sql.find_all(exp.Table) if _get_main_parent(t) == _get_main_parent(column)]
            for candidate_table in candidate_tables:
                table_columns = get_table_all_columns(db_path, candidate_table.name)
                if column_name.lower() in [col.lower() for col in table_columns]:
                    table_name = candidate_table.name
                    break

        if table_name:
            if table_name not in columns_dict:
                columns_dict[table_name] = []
            if column_name.lower() not in [c.lower() for c in columns_dict[table_name]]:
                columns_dict[table_name].append(column_name)

    return columns_dict

def _check_value_exists(db_path: str, table_name: str, column_name: str, value: str) -> Optional[str]:
    """
    Checks if a value exists in a column of a table in the database.
    
    Args:
        db_path (str): Path to the database file.
        table_name (str): The name of the table.
        column_name (str): The name of the column.
        value (str): The value to check.
        
    Returns:
        Optional[str]: The value if it exists, otherwise None.
    """
    query = f"SELECT `{column_name}` FROM {table_name} WHERE `{column_name}` LIKE '%{value}%' LIMIT 1"
    result = execute_sql(db_path, query, "one")
    return result[0] if result else None

def get_sql_condition_literals(db_path: str, sql: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Retrieves literals used in SQL query conditions and checks their existence in the database.
    
    Args:
        db_path (str): Path to the database file.
        sql (str): The SQL query string.

    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary of tables and their columns with condition literals.
    """
    try:
        columns_dict = get_sql_columns_dict(db_path=db_path, sql=sql)
        used_entities = {}
        for sql_exp in parse_one(sql, read="sqlite").flatten():
            for literal in sql_exp.find_all(exp.Literal):
                if literal == literal.parent.expression:
                    for column_exp in literal.parent.find_all(exp.Column):
                        column_name = column_exp.name
                        for table_name, column_names in columns_dict.items():
                            if column_name.lower() in [col.lower() for col in column_names]:
                                example_exist = False
                                example = literal.this
                                if "(" in str(literal.parent):
                                    value_check = _check_value_exists(db_path, table_name, column_name, literal.this)
                                    if value_check:
                                        example_exist = True
                                        example = value_check
                                if "LIKE" in str(literal.parent):
                                    example_to_search = literal.this.replace("%", "")
                                    value_check = _check_value_exists(db_path, table_name, column_name, example_to_search)
                                    if value_check:
                                        example_exist = True
                                        example = example_to_search
                                else:
                                    example_exist = True
                                if example_exist:
                                    if table_name not in used_entities:
                                        used_entities[table_name] = {}
                                    if column_name not in used_entities[table_name]:
                                        used_entities[table_name][column_name] = []
                                    if example not in used_entities[table_name][column_name]:
                                        used_entities[table_name][column_name].append(example)
        return used_entities
    except Exception as e:
        logging.critical(f"Error in get_sql_condition_literals: {e}\nSQL {sql}\n")
        return {}

def is_sql_select_distinct(sql: str, include_subqueries: bool = False) -> bool:
    """
    Checks whether the SQL uses DISTINCT at the SELECT level (e.g., SELECT DISTINCT ... or SELECT DISTINCT ON ...).

    Note:
        This checks the DISTINCT modifier of SELECT, and intentionally does NOT treat
        aggregate usage like COUNT(DISTINCT col) as "SELECT DISTINCT".

    Args:
        sql (str): The SQL query string.
        include_subqueries (bool): If True, returns True if any SELECT (including subqueries/CTEs)
            is DISTINCT. If False, only checks the outermost SELECT.
    """
    try:
        parsed_sql = parse_one(sql, read="sqlite")
        selects = list(parsed_sql.find_all(exp.Select))
        if not selects:
            return False
        if include_subqueries:
            return any(bool(sel.args.get("distinct")) for sel in selects)
        return bool(selects[0].args.get("distinct"))
    except Exception as e:
        logging.critical(f"Error in is_sql_select_distinct: {e}\nSQL: {sql}")
        raise e

def is_sql_use_limit(sql: str, include_subqueries: bool = False) -> bool:
    """
    Checks whether the SQL uses a LIMIT clause.

    Args:
        sql (str): The SQL query string.
        include_subqueries (bool): If True, returns True if any LIMIT exists anywhere in the query
            (including subqueries/CTEs). If False, only checks the outermost query's LIMIT.
    """
    try:
        parsed_sql = parse_one(sql, read="sqlite")
        if include_subqueries:
            return next(parsed_sql.find_all(exp.Limit), None) is not None
        return bool(parsed_sql.args.get("limit"))
    except Exception as e:
        logging.critical(f"Error in is_sql_use_limit: {e}\nSQL: {sql}")
        raise e
