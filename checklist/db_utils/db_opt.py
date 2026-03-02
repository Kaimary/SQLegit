import os, sqlite3

sqlite_reserved_keywords = [
    "ABORT", "ACTION", "ADD", "AFTER",
    "ALL", "ALTER", "ANALYZE", "AND",
    "AS", "ASC", "ATTACH", "AUTOINCREMENT",
    "BEFORE", "BEGIN", "BETWEEN", "BY",
    "CASCADE", "CASE", "CAST", "CHECK",
    "COLLATE", "COLUMN", "COMMIT", "CONFLICT",
    "CONSTRAINT", "CREATE", "CROSS", "CURRENT",
    "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
    "DATABASE", "DEFAULT", "DEFERRABLE", "DEFERRED",
    "DELETE", "DESC", "DETACH", "DISTINCT",
    "DROP", "EACH", "ELSE", "END",
    "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS",
    "EXPLAIN", "FAIL", "FOR", "FOREIGN",
    "FROM", "FULL", "GLOB", "GROUP",
    "HAVING", "IF", "IGNORE", "IMMEDIATE",
    "IN", "INDEX", "INDEXED", "INITIALLY",
    "INNER", "INSERT", "INSTEAD", "INTERSECT",
    "INTO", "IS", "ISNULL", "JOIN",
    "KEY", "LEFT", "LIKE", "LIMIT",
    "MATCH", "NATURAL", "NO", "NOT",
    "NOTNULL", "NULL", "OF", "OFFSET",
    "ON", "OR", "ORDER", "OUTER",
    "PLAN", "PRAGMA", "PRIMARY", "QUERY",
    "RAISE", "RECURSIVE", "REFERENCES", "REGEXP",
    "REINDEX", "RELEASE", "RENAME", "REPLACE",
    "RESTRICT", "RIGHT", "ROLLBACK", "ROW",
    "SAVEPOINT", "SELECT", "SET", "TABLE",
    "TEMP", "TEMPORARY", "THEN", "TO",
    "TRANSACTION", "TRIGGER", "UNION", "UNIQUE",
    "UPDATE", "USING", "VACUUM", "VALUES",
    "VIEW", "VIRTUAL", "WHEN", "WHERE",
    "WITH", "WITHOUT"
]

sqlite_type_map = {
    'INT': int,
    'INTEGER': int,
    'REAL': float,
    'TEXT': str,
    'BLOB': bytes,
    'NUMERIC': float,
    'DATE': str,
    'DATETIME': str,
    'bool': bool,
    "VARCHAR": str
}

def duplicate_sqlite_database(src_db_path, dest_db_path, reset=True):
    # 删除旧文件
    if os.path.exists(dest_db_path):
        os.remove(dest_db_path)
        
    src = sqlite3.connect(src_db_path)
    dest = sqlite3.connect(dest_db_path)
    # only duplicate schema
    if reset:
        query = "SELECT sql FROM sqlite_master WHERE type='table'"
        for (sql,) in src.execute(query):
            if sql and "sqlite_sequence" not in sql.lower():
                dest.execute(sql)
        dest.commit()
    else:
        src.backup(dest)
    src.close()
    dest.close()

# def duplicate_sqlite_database(src_db_path, dest_db_path, reset=False):
#     source = sqlite3.connect(src_db_path)
#     dest = sqlite3.connect(dest_db_path)

#     # Copy using the backup function
#     with dest:
#         source.backup(dest)
    
#     if reset: # Empty data instance
#         cursor = dest.cursor()
#         # Get all table names (excluding SQLite internal tables)
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
#         tables = cursor.fetchall()
#         # Disable foreign key checks temporarily to avoid constraint issues
#         cursor.execute("PRAGMA foreign_keys = OFF")
#         for (table_name,) in tables:
#             if table_name.upper() in sqlite_reserved_keywords: table_name = f'"{table_name}"'
#             cursor.execute(f"DELETE FROM {table_name}")
#         # Re-enable foreign keys
#         cursor.execute("PRAGMA foreign_keys = ON")
#         dest.commit()

#     source.close()
#     dest.close()
#     # print(f"Database copied from {os.path.basename(src_db_path)} to {dest_db_path}")
#     return

def insert_rows_into_table(db_path, table_name, rows):
    """
    Insert multiple rows into a specified table in an SQLite database.

    Parameters:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to insert data into.
        rows (list of tuples): List of data rows to insert. Each row should be a tuple.
    """
    if not rows: return "empty data"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Determine number of columns from the first row
        placeholders = ','.join('?' for _ in rows[0])
        if table_name.upper() in sqlite_reserved_keywords: table_name = f'"{table_name}"'
        sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

        cursor.executemany(sql, rows)
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return f"{e}"
    finally:
        conn.close()
    
    return None

def create_sqlite_database(db_path, schema_string):
    """
    Create an SQLite database with the given schema.

    Parameters:
        db_path (str): Path to the SQLite database file to create.
        schema_string (str): SQL schema string to define the database structure.
    """
    if os.path.exists(db_path):
        os.remove(db_path)  # Remove existing database file

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_string = schema_string.replace('--', '') # Remove comments to avoid execution issues
    try:
        cursor.executescript(schema_string)
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()