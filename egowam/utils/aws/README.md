# AWS Infrastructure
On AWS we leverage S3, RDS (SQL) and EC2 via Ray

## Logging into AWS CLI

```
Install aws cli
aws configure
It will prompt for "access key".  Get this on aws console IAM > Security Credentials > Access Keys
```

## Setting up RDS (ONE TIME SETUP, JUST FOR SIMAR'S REF)

```bash
sh egowam/utils/aws/create_rds.sh
sh egowam/utils/aws/create_rds_user.sh
sh egowam/utils/aws/create_schema.sh
```

## RDS Table Schema
See egowam/utils/aws/aws_sql.py:TableRow for schema
