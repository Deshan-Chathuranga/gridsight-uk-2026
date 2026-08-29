#!/bin/bash
# LocalStack AWS S3 bucket initialization script for local development

set -e

echo "Initializing LocalStack AWS S3 buckets..."

AWS_ENDPOINT="http://localhost:4566"
REGION="eu-west-2"

# Wait for LocalStack to be ready
until aws --endpoint-url=${AWS_ENDPOINT} s3 ls > /dev/null 2>&1; do
  echo "Waiting for LocalStack S3 emulator to start..."
  sleep 2
done

echo "LocalStack S3 is online. Creating default buckets..."

# Create buckets
aws --endpoint-url=${AWS_ENDPOINT} s3 mb s3://gridsight-data-bucket --region ${REGION} || true
aws --endpoint-url=${AWS_ENDPOINT} s3 mb s3://gridsight-artifacts-bucket --region ${REGION} || true

echo "Buckets created successfully:"
aws --endpoint-url=${AWS_ENDPOINT} s3 ls

echo "LocalStack AWS Initialization Complete!"
