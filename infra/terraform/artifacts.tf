# -----------------------------------------------------------------------------
# Model artifacts bucket for the GitHub Actions deploy job.
#
# The API image bakes in mlruns/, mlflow.db and the demand-history parquet,
# which are gitignored. They are uploaded here once after local training
# (scripts/aws/upload_artifacts.ps1 -Bucket <model_artifacts_bucket>) and the
# deploy job in .github/workflows/ci.yml pulls them before `docker build`.
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "model_artifacts" {
  bucket        = "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "model_artifacts_bucket" {
  description = "S3 bucket holding the baked model artifacts pulled by the deploy job."
  value       = aws_s3_bucket.model_artifacts.bucket
}
