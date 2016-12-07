import unittest

from data.tree import Node


class TestTree(unittest.TestCase):
    def setUp(self):
        self._root = Node("Node1", ["A", "B", "C"])

        self._c1 = Node("Node2", [], parent=self._root)
        self._c2 = Node("Node3", ["D"], parent=self._root)
        self._root.set_children_for_property("A", (self._c1, self._c2))

        self._c3 = Node("Node4", ["D"], parent=self._root)
        self._root.set_children_for_property("B", [self._c3])

        self._c4 = Node("Node5", [], parent=self._c3)
        self._c3.set_children_for_property("D", (self._c4))

        self._c5 = Node("Node6", [], parent=self._root)
        self._root.set_children_for_property("C", [self._c5])

    def test_location(self):
        self.assertEqual(self._c1.parent_and_pos(), (self._root, "A", 0))
        self.assertEqual(self._c2.parent_and_pos(), (self._root, "A", 1))
        self.assertEqual(self._c3.parent_and_pos(), (self._root, "B", 0))
        self.assertEqual(self._c4.parent_and_pos(), (self._c3, "D", 0))

    def test_preorder(self):
        preorder = [n for n in self._root]
        self.assertEqual(preorder, [self._root, self._c1, self._c2, self._c3, self._c4, self._c5])

    def test_maximal_size_subtree(self):
        n1 = Node("Node1", ("A", "B"))
        n2 = Node("Node2", ())
        n1.set_children_for_property("A", tuple(n2))
        n3 = Node("Node1", ("A", "B"))
        n1.set_children_for_property('B', tuple(n3))
        n4 = Node("Node3", ())
        n3.set_children_for_property('A', tuple(n4))
        n5 = Node("Node3", ())
        n3.set_children_for_property('B', tuple(n5))

        self.assertEqual(len(n1.maximal_common_subtree(n1)), 5)
        self.assertEqual(str(n1.maximal_common_subtree(n1)), str(n1))
        self.assertEqual(len(n3.maximal_common_subtree(n1)), 3)
        self.assertEqual(str(n3.maximal_common_subtree(n1)), str(n3))

        m1 = Node('Node2', ())
        self.assertEqual(len(m1.maximal_common_subtree(n1)), 1)
        self.assertEqual(len(m1.maximal_common_subtree(n2)), 1)
        self.assertEqual(m1.maximal_common_subtree(n3), None)

        m1 = Node("Node1", ("A", "B"))
        m2 = Node("Node4", ())
        m1.set_children_for_property('A', (m2))
        m3 = Node("Node3", ())
        m1.set_children_for_property('B', (m3))
        self.assertEqual(len(m1.maximal_common_subtree(n1)), 2)
        self.assertEqual(len(n1.maximal_common_subtree(m1)), 2)

    def test_print(self):
        pass


if __name__ == '__main__':
    unittest.main()
