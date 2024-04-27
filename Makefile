VENV_NAME = myenv
REQUIREMENTS = requirements.txt

help:
	@echo "Usage: make [COMMAND]"
	@echo
	@echo "Commands:"
	@echo "  install         Installs dependencies"
	@echo "  setup           Sets up a new virtual environment and activates it"
	@echo "  activate        Activates the virtual environment"
	@echo "  update          Updates the virtual environment and requirements"
	@echo "  add             Adds new dependency to requirements.txt"
	@echo "  clean           Cleans the virtual environment"
	@echo
	@echo "Example usage:"
	@echo "  make install"
	@echo "  make setup"
	@echo "  make activate"
	@echo "  make update"
	@echo "  make add requests"
	@echo "  make clean"

install:
	@echo "Installing dependencies..."
	python3 -m venv $(VENV_NAME)
	source $(VENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)
	@echo "Install complete."

setup:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV_NAME)
	source $(VENV_NAME)/bin/activate
	@echo "Setup complete."

activate:
	@echo "Activating virtual environment..."
	source $(VENV_NAME)/bin/activate
	@echo "Environment activated."

update:
	@echo "Updating virtual environment..."
	source $(VENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)
	@echo "Update complete."

add:
ifndef PACKAGE
	@echo "Usage: make add PACKAGE=<dependency>"
	@exit 1
endif
	@echo "Adding dependency $(PACKAGE) to requirements.txt..."
	echo $(PACKAGE) >> $(REQUIREMENTS)
	@echo "Dependency added."

clean:
	@echo "Cleaning virtual environment..."
	deactivate || true
	rm -rf $(VENV_NAME)
	@echo "Clean complete."
