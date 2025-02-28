module.exports = {
    apps: [{
      name: "gita-chatbot",
      script: "streamlit run bagavadgita.py",
      interpreter: "bash",
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        NODE_ENV: 'development',
        GROQ_API_KEY:"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",
        MILVUS_TOKEN:"99cd003de770782d436a049c87fb669188dc4424443531a325043d7f42859ca8c3d058b952d2e92d33677cf72b4931d12150c29d",
        MILVUS_URI:"https://in03-4e569f605c32eab.serverless.gcp-us-west1.cloud.zilliz.com"
      },
      env_production: {
        NODE_ENV: 'production',
        GROQ_API_KEY:"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",
        MILVUS_TOKEN:"99cd003de770782d436a049c87fb669188dc4424443531a325043d7f42859ca8c3d058b952d2e92d33677cf72b4931d12150c29d",
        MILVUS_URI:"https://in03-4e569f605c32eab.serverless.gcp-us-west1.cloud.zilliz.com"
      }
    }]
  };