#!/bin/sh
coverage erase
coverage run --branch --source=. -m pytest tests
coverage html -i