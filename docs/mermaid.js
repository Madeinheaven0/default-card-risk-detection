```mermaid
flowchart TD

    %% ==========================================
    %% NODE STYLES
    %% ==========================================

    classDef user fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    classDef frontend fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    classDef preprocessing fill:#1e1b4b,stroke:#c084fc,stroke-width:2px,color:#f8fafc
    classDef feast fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc
    classDef model fill:#451a03,stroke:#fbbf24,stroke-width:2px,color:#f8fafc
    classDef storage fill:#334155,stroke:#94a3b8,stroke-width:1px,color:#f8fafc

    %% ==========================================
    %% 1. OFFLINE PIPELINE
    %% Batch Processing, Feature Engineering and Training
    %% ==========================================

    subgraph OFFLINE["OFFLINE PIPELINE - Batch Processing and Model Training"]
        direction TB

        RawData[("Raw Data<br>data/ and make_dataset.py")]:::storage

        subgraph PrepPolars["Preprocessing - Polars Engine"]
            direction TB
            CleanPolars["clean_raw_data_polars()<br>Drop useless features"]:::preprocessing
            TransPolars["transform_features_polars()<br>Payment statistics: 3m, 12m, 24m<br>Delinquency score and financial ratios<br>Cleanliness index"]:::preprocessing
        end

        subgraph FeastOffline["Feast Feature Store - feature_repo/"]
            direction TB
            Entities["entities.py and features.py<br>create_timestamp.py<br>views.py and services.py"]:::feast
            OfflineStore[("Offline Store<br>Parquet and historical features")]:::storage
            OnlineStore[("Online Store<br>Materialized features for inference")]:::storage
        end

        subgraph Training["Model Training and Experiment Tracking"]
            direction TB
            Split["split_dataset()<br>Stratified Train / Validation / Test split"]:::preprocessing
            ColTrans["create_transformer()<br>Numerical: Median Imputer + StandardScaler<br>Categorical: Constant Imputer + OneHotEncoder"]:::preprocessing
            XGB(["XGBoost Classifier Model"]):::model
            MLflow[("MLflow Tracking and Registry<br>mlflow.db and mlruns/")]:::storage
        end

        %% Offline data flow

        RawData --> CleanPolars
        CleanPolars --> TransPolars

        TransPolars --> Entities
        Entities --> OfflineStore

        OfflineStore -->|"Feast materialize"| OnlineStore

        OfflineStore --> Split
        Split --> ColTrans
        ColTrans --> XGB

        XGB -->|"Log metrics and artifacts"| MLflow
    end

    %% ==========================================
    %% 2. ONLINE PIPELINE
    %% Real-Time Prediction and Inference
    %% ==========================================

    subgraph ONLINE["ONLINE PIPELINE - Application and Real-Time Inference"]
        direction TB

        User(["User"]):::user

        Streamlit["Web Interface<br>src/app/app_streamlit.py"]:::frontend

        subgraph FastAPI["FastAPI Backend - src/app/"]
            direction TB
            MainAPI["main.py and schemas.py<br>Pydantic input validation"]:::frontend
            Router["routers/prediction.py<br>Prediction endpoint"]:::frontend
        end

        subgraph PrepPandas["Preprocessing - Pandas Engine"]
            direction TB
            RatiosPandas["add_credit_financial_ratios_pandas()<br>Compute financial ratios on-the-fly<br>Clean Inf and NaN values"]:::preprocessing
        end

        ModelProd(["Loaded Production Model<br>MLflow / models/"]):::model

        Monitoring[("Monitoring and Logs<br>monitoring/ and logger.py")]:::storage

        %% User interaction

        User -->|"1. Enter client data"| Streamlit
        Streamlit -->|"2. HTTP POST - JSON"| MainAPI

        %% API processing

        MainAPI --> Router

        %% Historical feature retrieval

        Router <-->|"3. Retrieve historical features"| OnlineStore

        %% Feature engineering and inference

        Router -->|"4. Raw data and retrieved features"| RatiosPandas
        RatiosPandas -->|"5. Enriched features"| ModelProd

        %% Model loading

        MLflow -.->|"Load registered model"| ModelProd

        %% Prediction response

        ModelProd -->|"6. Default probability"| Router
        Router -->|"7. JSON response"| Streamlit
        Streamlit -->|"8. Display risk score"| User

        %% Monitoring

        Router -.->|"Log requests and predictions"| Monitoring
    end
```
