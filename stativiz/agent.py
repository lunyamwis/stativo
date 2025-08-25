import requests

def query_agent(query):
    """
    Query the agent with a specific query.
    
    
    :param query: The query string to send to the agent.
    :return: Response from the agent.
    """
    agent_url = "https://stativiz.com/api/metrics/agent/"
    response = requests.post(agent_url, json={"query": query}, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()



def query_firestore_agent(query):
    """
    Query the Firestore agent with a specific query.
    
    :param query: The natural language query string to send to the Firestore agent.
    :return: Response from the Firestore agent.
    """
    firestore_agent_url = "https://stativiz.com/api/firestore-agent/"
    response = requests.post(firestore_agent_url, json={"query": query}, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()