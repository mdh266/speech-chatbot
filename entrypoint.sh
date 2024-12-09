#!/usr/bin/env bash

streamlit run \
    main.py \
    --server.port 8080 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableXsrfProtection false \
    --server.enableCORS false
