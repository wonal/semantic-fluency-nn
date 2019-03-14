class IRT:
    @classmethod
    def calculate(cls, path):
        visited = set()
        count = 0
        redundant_visitations = []
        prev_unique_node = None
        for node in path:
            if node in visited:
                count += 1
            else:
                redundant_visitations.append(tuple((prev_unique_node, node, count)))
                visited.add(node)
                prev_unique_node = node
                count = 0
        return redundant_visitations

