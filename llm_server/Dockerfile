FROM ollama/ollama:latest

# Copy the start script into the container
COPY start.sh /start.sh

# Make the script executable
RUN chmod +x /start.sh

# Set the start script as the entry point
ENTRYPOINT ["/start.sh"]