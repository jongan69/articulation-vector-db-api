#!/bin/bash

# Test script for Articulation Vector DB API
# Base URL
BASE_URL="https://articulation-vector-db-api.onrender.com"

echo "🧪 Testing Articulation Vector DB API"
echo "======================================"
echo ""

# Test 1: Health Check (Root)
echo "1️⃣ Testing GET / (Health Check)"
echo "--------------------------------"
curl -X GET "${BASE_URL}/" \
  -H "Content-Type: application/json" \
  -w "\n\nStatus: %{http_code}\n\n"
echo ""

# Test 2: Detailed Health Check
echo "2️⃣ Testing GET /health"
echo "--------------------------------"
curl -X GET "${BASE_URL}/health" \
  -H "Content-Type: application/json" \
  -w "\n\nStatus: %{http_code}\n\n"
echo ""

# Test 3: Get Stats
echo "3️⃣ Testing GET /stats"
echo "--------------------------------"
curl -X GET "${BASE_URL}/stats" \
  -H "Content-Type: application/json" \
  -w "\n\nStatus: %{http_code}\n\n"
echo ""

# Test 4: Search Query
echo "4️⃣ Testing POST /search"
echo "--------------------------------"
curl -X POST "${BASE_URL}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the articulation agreements for University of Florida?",
    "top_k": 3
  }' \
  -w "\n\nStatus: %{http_code}\n\n"
echo ""

# Test 5: Query (alias endpoint)
echo "5️⃣ Testing POST /query"
echo "--------------------------------"
curl -X POST "${BASE_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transfer credits",
    "top_k": 2
  }' \
  -w "\n\nStatus: %{http_code}\n\n"
echo ""

# Test 6: Ingest PDFs (if PDFs are available on server)
echo "6️⃣ Testing POST /ingest"
echo "--------------------------------"
curl -X POST "${BASE_URL}/ingest" \
  -H "Content-Type: application/json" \
  -w "\n\nStatus: %{http_code}\n\n"
echo ""

echo "✅ Testing complete!"

