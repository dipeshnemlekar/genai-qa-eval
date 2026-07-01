def pytest_addoption(parser):
    parser.addini('human_eval', 'Enable or disable human evaluation (true/false/True/False)')
    parser.addini('auto_dashboard', 'Enable or disable automatic dashboard generation')
