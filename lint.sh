#!/bin/bash

# Print with colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running linters and formatters...${NC}\n"

# Run formatters first
echo -e "${GREEN}Running Black...${NC}"
black . || { echo -e "${RED}Black failed${NC}"; exit 1; }

echo -e "\n${GREEN}Running isort...${NC}"
isort . || { echo -e "${RED}isort failed${NC}"; exit 1; }

# Run linters
echo -e "\n${GREEN}Running flake8...${NC}"
flake8 . || { echo -e "${RED}flake8 failed${NC}"; exit 1; }

echo -e "\n${GREEN}Running pylint...${NC}"
pylint **/*.py || { echo -e "${RED}pylint failed${NC}"; exit 1; }

echo -e "\n${GREEN}All checks passed! ✨${NC}" 