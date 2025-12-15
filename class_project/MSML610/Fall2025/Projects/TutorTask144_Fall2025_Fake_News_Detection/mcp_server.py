"""
MCP REST API Server for BERT Fake News Detection

Exposes the BERT fake news classifier through REST API endpoints.
"""

import logging
import os
import sys
from flask import Flask, request, jsonify, render_template
from mcp_server_class import BERTMCPServer
import bert_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

mcp_server = None


def init_server():
    global mcp_server

    try:
        logger.info("Loading BERT model...")
        model, tokenizer = bert_utils.load_model('models/bert_fake_news')

        logger.info("Initializing MCP Server...")
        mcp_server = BERTMCPServer(model, tokenizer)

        logger.info("MCP Server initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MCP Server: {e}")
        return False


@app.route('/models', methods=['GET'])
def list_models():
    if mcp_server is None:
        return jsonify({'error': 'Server not initialized'}), 500

    try:
        models = mcp_server.list_models()
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    if mcp_server is None:
        return jsonify({'error': 'Server not initialized'}), 500

    try:
        info = mcp_server.get_model_info(model_id)
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if mcp_server is None:
        return jsonify({'error': 'Server not initialized'}), 500

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400

        model_id = data.get('model_id', 'bert_fake_news')
        text = data.get('text')

        result = mcp_server.predict(text)
        result['model_id'] = model_id

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    if mcp_server is None:
        return jsonify({'error': 'Server not initialized'}), 500

    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing required field: texts'}), 400

        model_id = data.get('model_id', 'bert_fake_news')
        texts = data.get('texts')

        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400

        result = mcp_server.predict_batch(texts)
        result['model_id'] = model_id

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/statistics', methods=['GET'])
def get_statistics():
    if mcp_server is None:
        return jsonify({'error': 'Server not initialized'}), 500

    try:
        stats = mcp_server.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    if mcp_server is None:
        return jsonify({'status': 'unhealthy', 'error': 'Server not initialized'}), 503

    return jsonify({'status': 'healthy', 'server': 'MCP Server running'})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if mcp_server is None:
        return jsonify({'error': 'Server not initialized', 'status': 'error'}), 500

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text', 'status': 'error'}), 400

        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Text field cannot be empty', 'status': 'error'}), 400

        result = mcp_server.predict(text)

        if 'error' in result:
            return jsonify({'error': result['error'], 'status': 'error'}), 500

        return jsonify({
            'status': 'success',
            'label': result['prediction']['class'],
            'confidence': result['prediction']['confidence'],
            'confidence_percent': result['prediction']['confidence_percent'],
            'processing_time_ms': result['metadata']['processing_time_ms'],
            'text_length': result['metadata']['text_length']
        })
    except Exception as e:
        logger.error(f"Error in web API prediction: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    # Initialize server
    if not init_server():
        logger.error("Failed to initialize MCP Server")
        sys.exit(1)

    # Get configuration from environment
    host = os.getenv('MCP_HOST', '0.0.0.0')
    port = int(os.getenv('MCP_PORT', 9090))
    debug = os.getenv('MCP_DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting MCP REST API Server on {host}:{port}")

    # Run Flask app
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
