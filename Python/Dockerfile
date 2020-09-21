# Base image
FROM pytorch/pytorch

# Make new directory for application
RUN mkdir /application
WORKDIR /application

# Install some dependencies
RUN pip install torchtext
RUN pip install boto3

# Copy src directory contents into application
COPY /src .

# Run docker test file
CMD ["python", "./Train_Eval_Prod.py"]