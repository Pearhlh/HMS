#!/bin/bash
set -e

# Modify pg_hba.conf to allow connections from all Docker containers
echo "host all all 0.0.0.0/0 trust" >> /var/lib/postgresql/data/pg_hba.conf

# Restart PostgreSQL to apply the changes
pg_ctl reload
