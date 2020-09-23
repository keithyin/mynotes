struct Node {
    int value;
    bool is_red = true;
    Node* left = nullptr;  // unique_ptr ?
    Node* right = nullptr;
    Node(int val) : value(val) {}
};

class BlackRedTree {
   public:
    Node* Put(Node* father, int val) {
        if (father == nullptr) {
            return new Node(val);
        }
        if (father->value < val) father->left = Put(father->left, val);
        if (father->value > val) father->right = Put(father->right, val);

        if (IsRed(father->right) && !IsRed(father->left)) father = RotateLeft(father);
        if (IsRed(father->left) && IsRed(father->left->left)) father = RotateRight(father);
        if (IsRed(father->left) && IsRed(father->right)) ColorFlip(father);
        return father;
    }

   private:
    Node* RotateLeft(Node* node) {
        Node* tmp = node->right;
        node->right = node->right->left;
        tmp->left = node;
        tmp->is_red = false;
        tmp->left->is_red = true;
        return tmp;
    }
    void ColorFlip(Node* node) {
        node->is_red = true;
        node->left->is_red = false;
        node->right->is_red = false;
    }

    Node* RotateRight(Node* node) {
        Node* tmp = node->left;
        node->left = tmp->right;
        tmp->right = node;
        node->is_red = true;
        tmp->is_red = false;
        return tmp;
    }

    bool IsRed(Node* node) {
        if (node == nullptr) return false;
        return node->is_red;
    }
};
