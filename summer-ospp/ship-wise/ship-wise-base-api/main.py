import os

from app import create_app

os.chdir(os.path.abspath(os.path.dirname(__file__)))
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=8100)
