# AWS Free Tier & S3 Setup Guide for GridSight UK

This guide walks you step-by-step through setting up a **100% Free AWS Free Tier** S3 bucket for storing GridSight UK model checkpoints, parquet feature datasets, and fan plot diagnostics.

---

## Step 1: Create an AWS Free Tier Account

1. Go to [https://aws.amazon.com/free/](https://aws.amazon.com/free/).
2. Click **Create an AWS Account**.
3. AWS Free Tier provides **5 GB of S3 standard storage** and **20,000 GET / 2,000 PUT requests per month for 12 months** free of charge.

---

## Step 2: Create an S3 Bucket

1. Sign in to the [AWS Management Console](https://console.aws.amazon.com/).
2. Search for **S3** in the search bar.
3. Click **Create bucket**.
4. Configure bucket settings:
   - **Bucket name**: e.g., `gridsight-uk-storage` (Must be globally unique across all AWS users).
   - **AWS Region**: Select `eu-west-2 (London)` (or your preferred region).
   - **Object Ownership**: ACLs disabled (recommended).
   - **Block Public Access settings**: Keep *Block all public access* checked (recommended for security).
5. Click **Create bucket**.

---

## Step 3: Create IAM Access Keys for GridSight UK

For security, create an IAM user rather than using root credentials:

1. Search for **IAM** in the AWS Console.
2. Click **Users** ➔ **Add user**.
3. User name: `gridsight-app-user`.
4. Click **Next** ➔ Select **Attach policies directly**.
5. Search for and select: `AmazonS3FullAccess` (or create a custom policy restricting to your bucket).
6. Click **Next** ➔ **Create user**.
7. Click on the newly created user `gridsight-app-user`.
8. Go to the **Security credentials** tab.
9. Scroll to **Access keys** ➔ Click **Create access key**.
10. Select **Application running outside AWS** ➔ Click **Next** ➔ **Create access key**.
11. Copy your:
    - **Access key ID** (e.g. `AKIAIOSFODNN7EXAMPLE`)
    - **Secret access key** (e.g. `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`)

---

## Step 4: Configure Local `.env` File

Add your AWS credentials to your `.env` file in the project root:

```ini
# AWS S3 Cloud Storage Credentials
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=eu-west-2
AWS_S3_BUCKET=gridsight-uk-storage
```

> [!NOTE]
> If `AWS_ENDPOINT_URL` is empty or not set, GridSight automatically connects directly to AWS S3. If `AWS_ENDPOINT_URL=http://localhost:4566` is set, it routes to LocalStack local emulator.

---

## Step 5: Configure GitHub Secrets for Railway & AWS Deployment

To enable GitHub Actions CD pipeline to automatically test, build, and deploy your code:

1. Go to your GitHub repository on web browser.
2. Click **Settings** ➔ **Secrets and variables** ➔ **Actions**.
3. Add the following **Repository secrets**:
   - `RAILWAY_TOKEN`: Your Railway project deployment token (from Railway dashboard settings).
   - `AWS_ACCESS_KEY_ID`: Your IAM access key ID.
   - `AWS_SECRET_ACCESS_KEY`: Your IAM secret access key.
   - `AWS_S3_BUCKET`: `gridsight-uk-storage`.

---

## Step 6: Test Connection via FastAPI API

Start the backend server:

```bash
uvicorn apps.backend.app:app --reload --port 7860
```

Open Swagger docs at `http://localhost:7860/docs` and test:
- `GET /api/storage/status`: Should display `"connected": true` and `"mode": "AWS Cloud S3"`.
- `GET /api/storage/artifacts`: Lists files in your AWS S3 bucket.
