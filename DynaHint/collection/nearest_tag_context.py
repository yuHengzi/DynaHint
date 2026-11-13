"""Read-only PostgreSQL statistics snapshots for execution-free TAG features."""
import csv


def _array(value):
    return next(csv.reader([value[1:-1]], escapechar='\\')) if value and value != '{}' else []


def collect_context(cursor, dataset):
    cursor.execute("SELECT c.relname,greatest(c.reltuples,0),c.relpages,coalesce(s.n_dead_tup,0),coalesce(s.n_mod_since_analyze,0) "
                   "FROM pg_class c LEFT JOIN pg_stat_user_tables s ON s.relid=c.oid "
                   "WHERE c.relnamespace='public'::regnamespace AND c.relkind='r'")
    tables = {r[0]: list(r[1:]) for r in cursor.fetchall()}
    cursor.execute("SELECT tablename,attname,null_frac,n_distinct,histogram_bounds::text,most_common_vals::text,most_common_freqs "
                   "FROM pg_stats WHERE schemaname='public'")
    columns = {table+'.'+column: dict(null_frac=nulls, ndv=ndv, histogram=_array(hist),
                                     mcv=_array(common), frequencies=freq or [])
               for table,column,nulls,ndv,hist,common,freq in cursor.fetchall()}
    dates = None
    if dataset.upper() == 'TPCDS':
        cursor.execute('SELECT d_year,min(d_date_sk),max(d_date_sk) FROM date_dim GROUP BY d_year')
        dates = dict(years={str(y): [lo,hi] for y,lo,hi in cursor.fetchall()}, columns={})
        for table,column in {'store_sales':'ss_sold_date_sk','catalog_sales':'cs_sold_date_sk',
                             'web_sales':'ws_sold_date_sk','inventory':'inv_date_sk',
                             'store_returns':'sr_returned_date_sk','catalog_returns':'cr_returned_date_sk',
                             'web_returns':'wr_returned_date_sk'}.items():
            value = columns.get(table+'.'+column)
            if value is not None:
                dates['columns'][table] = dict(value, histogram=[float(x) for x in value['histogram']],
                                               mcv=[float(x) for x in value['mcv']])
    return tables, columns, dates
