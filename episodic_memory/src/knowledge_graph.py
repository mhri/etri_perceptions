from neo4j.v1 import GraphDatabase, basic_auth
from py2neo import Graph, Node, Relationship

from datetime import datetime

class KnowledgeGraph:
	def __init__(self):
		self.driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j","Cog1234"))
		self.session = self.driver.session()

		# py2neo driver
		self.graph = Graph(password="Cog1234")

	def Run(self, query):
		result = self.session.run(query)
		return result

	def Create(self, node_type, prop, value):
		query = "CREATE (a:{0} {{{1}:\'{2}\'}})".format(node_type,prop,value)
		#print "Create: ", query
		self.session.run(query)

	def Read(self, node_type, ref_key, ref_value, key):
		query = "MATCH (x:{0}) WHERE x.{1}=\'{2}\' RETURN x.{3} AS {3}".format(node_type,ref_key,ref_value,key)
		#print "Read: ", query
		result = self.session.run(query)
		value = None
		for record in result:
			#print "OK:", record[key]
			value = record[key]
		return value

	def Set(self, node_type, ref_key, ref_value, key, value):
		query = "MATCH (x:{0}) WHERE x.{1} = \'{2}\' SET x.{3} = \'{4}\' RETURN x".format(node_type,ref_key,ref_value,key,value)
		#print "Set: ", query
		result = self.session.run(query)
		#print "   Result: ", result

	def Delete(self, node_type, ref_key, ref_value):
		query = "MATCH (x:{0}) WHERE x.{1} = \'{2}\' DELETE x".format(node_type,ref_key,ref_value)
		#print "Delete: ", query
		self.session.run(query)

	def AddPercept(self, pid, props):
		person = self.graph.find_one("PerceptedPerson", property_key="id", property_value=pid)

		tx = self.graph.begin()
		percept = Node("Percept", timestamp=str(datetime.now()), next="none")
		tx.create(percept)
		for key in props:
			val = props[key]
			n = Node("Value", value=val)
			rel = Relationship(percept, key, n)
			tx.create(n)
			print 'creating... : ', rel
			tx.create(rel)

		r = None
		if person is None:
			person = Node("PerceptedPerson", id=pid)
			tx.create(person)
			rel = Relationship(person, "first_percept", percept)
			tx.create(rel)
		else:
			r = self.graph.match_one(start_node=person, rel_type="last_percept")
			rel = Relationship(r.end_node(), 'next', percept)
			tx.create(rel)
			query = "MATCH (p:Person)-[r:last_percept]->(:Percept) WHERE p.id=\'{0}\' DELETE r".format(person['id'])
			print query
			tx.run(query)

		rel = Relationship(person, 'last_percept', percept)
		tx.create(rel)
		rel = Relationship(person, 'percept', percept)
		tx.create(rel)
		tx.commit()
