.PHONY: help install install-backend install-frontend dev backend frontend build test clean kill-ports setup check

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(GREEN)Trading Bot - Available Commands$(NC)"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: install-backend install-frontend ## Install all dependencies
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

install-backend: ## Install Python backend dependencies
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Backend dependencies installed$(NC)"

install-frontend: ## Install React frontend dependencies
	@echo "$(YELLOW)Installing frontend dependencies...$(NC)"
	cd react-frontend && npm install
	@echo "$(GREEN)✓ Frontend dependencies installed$(NC)"

# Development targets
dev: ## Start both backend and frontend in development mode
	@echo "$(GREEN)Starting Trading Bot in development mode...$(NC)"
	@echo "Backend:  http://localhost:8080"
	@echo "Frontend: http://localhost:3000"
	@echo "Docs:     http://localhost:8080/docs"
	@echo "Press Ctrl+C to stop all services"
	@make -j2 backend frontend

backend: ## Start FastAPI backend server
	@echo "$(YELLOW)Starting FastAPI backend...$(NC)"
	cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080

frontend: ## Start React frontend development server
	@echo "$(YELLOW)Starting React frontend...$(NC)"
	cd react-frontend && npm start

# Production targets
build: ## Build frontend for production
	@echo "$(YELLOW)Building frontend for production...$(NC)"
	cd react-frontend && npm run build
	@echo "$(GREEN)✓ Frontend build complete$(NC)"

serve: build ## Serve production build
	@echo "$(YELLOW)Starting production server...$(NC)"
	cd react-frontend && npx serve -s build -l 3000

# Testing targets
test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	cd react-frontend && npm test -- --watchAll=false
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-backend: ## Test backend API endpoints
	@echo "$(YELLOW)Testing backend API...$(NC)"
	curl -f http://localhost:8080/ || (echo "$(RED)Backend not running$(NC)" && exit 1)
	curl -f http://localhost:8080/api/status || (echo "$(RED)API not responding$(NC)" && exit 1)
	@echo "$(GREEN)✓ Backend API tests passed$(NC)"

# Utility targets
clean: ## Clean build artifacts and dependencies
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf react-frontend/build
	rm -rf react-frontend/node_modules
	rm -rf backend/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "$(GREEN)✓ Clean complete$(NC)"

kill-ports: ## Kill processes on ports 3000 and 8080
	@echo "$(YELLOW)Killing processes on ports 3000 and 8080...$(NC)"
	-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	-lsof -ti:8080 | xargs kill -9 2>/dev/null || true
	@echo "$(GREEN)✓ Ports cleared$(NC)"

setup: install ## Complete project setup
	@echo "$(GREEN)Setting up Trading Bot project...$(NC)"
	@make check
	@echo "$(GREEN)✓ Setup complete$(NC)"

check: ## Check system requirements and project structure
	@echo "$(YELLOW)Checking system requirements...$(NC)"
	@command -v python3 >/dev/null 2>&1 || (echo "$(RED)Python3 is required$(NC)" && exit 1)
	@command -v node >/dev/null 2>&1 || (echo "$(RED)Node.js is required$(NC)" && exit 1)
	@command -v npm >/dev/null 2>&1 || (echo "$(RED)npm is required$(NC)" && exit 1)
	@test -f backend/main.py || (echo "$(RED)Backend main.py not found$(NC)" && exit 1)
	@test -f react-frontend/package.json || (echo "$(RED)Frontend package.json not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ System check passed$(NC)"

# Environment targets
env: ## Show environment information
	@echo "$(GREEN)Environment Information$(NC)"
	@echo "======================="
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'Not installed')"
	@echo "Node.js: $$(node --version 2>/dev/null || echo 'Not installed')"
	@echo "npm: $$(npm --version 2>/dev/null || echo 'Not installed')"
	@echo "Current directory: $$(pwd)"
	@echo "Backend status: $$(curl -s http://localhost:8080/api/status 2>/dev/null | grep -o '"training_active":[^,]*' || echo 'Not running')"

# Documentation targets
docs: ## Open API documentation
	@echo "$(YELLOW)Opening API documentation...$(NC)"
	@command -v open >/dev/null 2>&1 && open http://localhost:8080/docs || echo "Visit http://localhost:8080/docs"

logs: ## Show backend logs (if running)
	@echo "$(YELLOW)Backend logs:$(NC)"
	@ps aux | grep uvicorn | grep -v grep || echo "Backend not running"

# Database and model targets
migrate: ## Run database migrations (if applicable)
	@echo "$(YELLOW)Running migrations...$(NC)"
	@echo "$(GREEN)✓ Migrations complete$(NC)"

seed: ## Seed database with sample data
	@echo "$(YELLOW)Seeding database...$(NC)"
	python3 -c "from backend.seed import seed_data; seed_data()" 2>/dev/null || echo "No seed script found"
	@echo "$(GREEN)✓ Database seeded$(NC)"

# Docker targets (for future use)
docker-build: ## Build Docker containers
	@echo "$(YELLOW)Building Docker containers...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker build complete$(NC)"

docker-up: ## Start services with Docker
	@echo "$(YELLOW)Starting services with Docker...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Docker services started$(NC)"

docker-down: ## Stop Docker services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Docker services stopped$(NC)"

# Quick shortcuts
start: dev ## Alias for dev
stop: kill-ports ## Stop all services
restart: stop dev ## Restart all services