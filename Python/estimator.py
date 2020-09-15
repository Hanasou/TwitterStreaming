from sagemaker.pytorch import PyTorch

hyperparameters = {
    'epochs': 10,
    'batch_size': 100,
    'emb_dim': 50,
    'hidden_dim': 50,
    'num_classes': 3,
    'num_layers': 3,
    'lstm_units': 128,
    'num_hidden': 256,
    'lr': 0.001,
    'use_cuda': True
}

pytorch_estimator = PyTorch('sagemaker-train-deploy.py',
                            instance_type='ml.p3.2xlarge',
                            instance_count=1,
                            framework_version='1.6.0',
                            py_version='py3',
                            source_dir='code',
                            hyperparameters=hyperparameters)

pytorch_estimator.fit({'train': 's3://zhang-roy-sent140-training/training_processed_3.csv'})

# Deploy my estimator to a SageMaker Endpoint and get a Predictor
predictor = pytorch_estimator.deploy(instance_type='ml.m4.xlarge',
                                     initial_instance_count=1)

# `data` is a NumPy array or a Python list.
# `response` is a NumPy array.
response = predictor.predict(data)