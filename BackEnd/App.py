import subprocess
import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

# ─── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Home route ────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        'message': 'Cricket Team Prediction API',
        'endpoints': {
            '/predict/<match_type>': 'GET - match_type = test, odi, or t20',
            '/health': 'GET - health check'
        }
    })

# ─── Health check ───────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# ─── Prediction route ───────────────────────────────────────────────────────────
@app.route('/predict/<match_type>')
def predict_team(match_type):
    match_type = match_type.lower()
    script_mapping = {
        'test': 'Test_main.py',
        'odi': 'ODI_main.py',
        't20': 'T20_main.py'
    }

    if match_type not in script_mapping:
        return jsonify({
            'success': False,
            'error': "Invalid match type. Use 'test', 'odi', or 't20'."
        }), 400

    # Ensure we look in the same folder as App.py
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    script_name = script_mapping[match_type]
    script_path = os.path.join(base_dir, script_name)

    if not os.path.isfile(script_path):
        return jsonify({
            'success': False,
            'error': f"Prediction script not found: {script_name}"
        }), 500

    try:
        # Run the external script
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse its JSON output
        output = result.stdout.strip()
        predicted_team = json.loads(output)

        return jsonify({
            'success': True,
            'match_type': match_type.upper(),
            'team': predicted_team,
            'total_players': len(predicted_team),
            'batters': len([p for p in predicted_team if p['category'] == 'Batter']),
            'bowlers': len([p for p in predicted_team if p['category'] == 'Bowler'])
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            'success': False,
            'error': 'Script execution failed',
            'stderr': e.stderr
        }), 500

    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'error': 'Invalid JSON output from script',
            'raw_stdout': result.stdout
        }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ─── Run the server ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
