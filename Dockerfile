FROM public.ecr.aws/lambda/python:3.10

# Set working directory to Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy only requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --target '${LAMBDA_TASK_ROOT}'

# Copy required files
COPY . .

# Run the application
CMD ["api.handler"]
