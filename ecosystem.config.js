module.exports = { apps: [ { 
    name: "pdfsearch", 
    script: "streamlit", 
    args: "run pdfsearch.py", 
    env: { 
        GROQ_API_KEY:"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",
        MILVUS_TOKEN:"99cd003de770782d436a049c87fb669188dc4424443531a325043d7f42859ca8c3d058b952d2e92d33677cf72b4931d12150c29d",
        MILVUS_URI:"https://in03-4e569f605c32eab.serverless.gcp-us-west1.cloud.zilliz.com"
    }, 
}, ], };