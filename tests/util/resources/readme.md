# Setup Test Resources

To create a SQLite database, do the following:

1) run `crawl-fs` with the `--gen-sql` option, it will emit a `<basename>.sql` file alongside the database
2) edit the generated sql file as desired
3) place the edited sql file in this directory
4) run `sqlite3 ./test.db <outdir.basename.sql` to generate the test database