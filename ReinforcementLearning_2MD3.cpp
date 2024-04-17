#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stack>
#include <vector>

#include "cartCentering.h"

using namespace std;
typedef string Elem;

// returns a double unifomrly sampled in (0,1)
double randDouble(mt19937& rng) {
  return std::uniform_real_distribution<>{0, 1}(rng);
}
// returns uniformly sampled 0 or 1
bool randChoice(mt19937& rng) {
  return std::uniform_int_distribution<>{0, 1}(rng);
}
// returns a random integer uniformly sampled in (min, max)
int randInt(mt19937& rng, const int& min, const int& max) {
  return std::uniform_int_distribution<>{min, max}(rng);
}
// returns a random operator in type Elem
Elem randOperator(mt19937& rng) {
  vector<Elem> operators = {"+","-","*","/",">","abs"};
  return operators[randInt(rng,0,operators.size()-1)];
}
// returns a random operand (a, b, or a random constant)
Elem randOperand(mt19937& rng) {
  double randNum = randDouble(rng) * randInt(rng,-10,10); // randInt inputs choosen arbitrarily
  vector<string> operands = {"a","b",to_string(randNum)};
  return operands[randInt(rng,0,operands.size()-1)];
}
// returns a random operator or operand
Elem randElem(mt19937& rng) {
  return (randChoice(rng) == 1) ? randOperator(rng) : randOperand(rng);
}

// return true if op is a suported operation, otherwise return false
bool isOp(string op) {
  if (op == "+")
    return true;
  else if (op == "-")
    return true;
  else if (op == "*")
    return true;
  else if (op == "/")
    return true;
  else if (op == ">")
    return true;
  else if (op == "abs")
    return true;
  else
    return false;
}
// checks is the operator requires one or two operand(s)
int arity(string op) {
  if (op == "abs")
    return 1;
  else
    return 2;
}

class LinkedBinaryTree {
 public:
  struct Node {
    Elem elt;
    string name;
    Node* par;
    Node* left;
    Node* right;
    Node() : elt(), par(NULL), name(""), left(NULL), right(NULL) {}
    int depth() {
      if (par == NULL) return 0;
      return par->depth() + 1;
    }
  };

  class Position {
   private:
    Node* v;

   public:
    Position(Node* _v = NULL) : v(_v) {}
    Elem& operator*() { return v->elt; }
    Position left() const { return Position(v->left); }
    void setLeft(Node* n) { v->left = n; }
    Position right() const { return Position(v->right); }
    void setRight(Node* n) { v->right = n; }
    Position parent() const  // get parent
    {
      return Position(v->par);
    }
    bool isRoot() const  // root of the tree?
    {
      return v->par == NULL;
    }
    bool isExternal() const  // an external node?
    {
      return v->left == NULL && v->right == NULL;
    }
    friend class LinkedBinaryTree;  // give tree access
  };
  typedef vector<Position> PositionList;

 public:
  LinkedBinaryTree() : _root(NULL), score(0), steps(0), generation(0) {}

  // copy constructor
  LinkedBinaryTree(const LinkedBinaryTree& t) {
    _root = copyPreOrder(t.root());
    score = t.getScore();
    steps = t.getSteps();
    generation = t.getGeneration();
  }

  // copy assignment operator
  LinkedBinaryTree& operator=(const LinkedBinaryTree& t) {
    if (this != &t) {
      // if tree already contains data, delete it
      if (_root != NULL) {
        PositionList pl = positions();
        for (auto& p : pl) delete p.v;
      }
      _root = copyPreOrder(t.root());
      score = t.getScore();
      steps = t.getSteps();
      generation = t.getGeneration();
    }
    return *this;
  }

  // destructor
  ~LinkedBinaryTree() {
    if (_root != NULL) {
      PositionList pl = positions();
      for (auto& p : pl) delete p.v;
    }
  }

  int size() const { return size(_root); }
  int size(Node* root) const;
  int depth() const;
  bool empty() const { return size() == 0; };
  Node* root() const { return _root; }
  PositionList positions() const;
  void addRoot() { _root = new Node; }
  void addRoot(Elem e) {
    _root = new Node;
    _root->elt = e;
  }
  void addRoot(Node* n) {_root = copyPreOrder(n); }
  void nameRoot(string name) { _root->name = name; }
  void addLeftChild(const Position& p, const Node* n);
  void addLeftChild(const Position& p);
  void addRightChild(const Position& p, const Node* n);
  void addRightChild(const Position& p);
  void printExpression() { printExpression(_root); }
  void printExpression(Node* v);
  double evaluateExpression(double a, double b) {
    return evaluateExpression(Position(_root), a, b);
  };
  double evaluateExpression(const Position& p, double a, double b);
  long getGeneration() const { return generation; }
  void setGeneration(int g) { generation = g; }
  double getScore() const { return score; }
  void setScore(double s) { score = s; }
  double getSteps() const { return steps; }
  void setSteps(double s) { steps = s; }
  void randomExpressionTree(const Position& p, const int& maxDepth, mt19937& rng);
  void randomExpressionTree(const int& maxDepth, mt19937& rng) {
    randomExpressionTree(positions()[0], maxDepth, rng);
  }
  void deleteSubtree(Node* del);
  void deleteSubtreeMutator(mt19937& rng);
  void addSubtreeMutator(mt19937& rng, const int maxDepth);
  void crossover(LinkedBinaryTree& otherTree, mt19937& rng);

 protected:                                        // local utilities
  void preorder(Node* v, PositionList& pl) const;  // preorder utility
  Node* copyPreOrder(const Node* root);
  double score;     // mean reward over 20 episodes
  double steps;     // mean steps-per-episode over 20 episodes
  long generation;  // which generation was tree "born"
 private:
  Node* _root;  // pointer to the root
};

// add the tree rooted at node child as this tree's left child
void LinkedBinaryTree::addLeftChild(const Position& p, const Node* child) {
  Node* v = p.v;
  v->left = copyPreOrder(child);  // deep copy child
  v->left->par = v;
}

// add the tree rooted at node child as this tree's right child
void LinkedBinaryTree::addRightChild(const Position& p, const Node* child) {
  Node* v = p.v;
  v->right = copyPreOrder(child);  // deep copy child
  v->right->par = v;
}

// add an empty left child node to position p
void LinkedBinaryTree::addLeftChild(const Position& p) {
  Node* v = p.v;
  v->left = new Node;
  v->left->par = v;
}

// add an empty right child node to position p
void LinkedBinaryTree::addRightChild(const Position& p) {
  Node* v = p.v;
  v->right = new Node;
  v->right->par = v;
}

// return a list of all nodes
LinkedBinaryTree::PositionList LinkedBinaryTree::positions() const {
  PositionList pl;
  preorder(_root, pl);
  return PositionList(pl);
}

void LinkedBinaryTree::preorder(Node* v, PositionList& pl) const {
  pl.push_back(Position(v));
  if (v->left != NULL) preorder(v->left, pl);
  if (v->right != NULL) preorder(v->right, pl);
}

int LinkedBinaryTree::size(Node* v) const {
  if (v == NULL) return 0; // added to ensure functionality when tree is empty
  int lsize = 0;
  int rsize = 0;
  if (v->left != NULL) lsize = size(v->left);
  if (v->right != NULL) rsize = size(v->right);
  return 1 + lsize + rsize;
}

int LinkedBinaryTree::depth() const {
  PositionList pl = positions();
  int depth = 0;
  for (auto& p : pl) depth = std::max(depth, p.v->depth());
  return depth;
}

LinkedBinaryTree::Node* LinkedBinaryTree::copyPreOrder(const Node* root) {
  if (root == NULL) return NULL;
  Node* nn = new Node;
  nn->elt = root->elt;
  nn->left = copyPreOrder(root->left);
  if (nn->left != NULL) nn->left->par = nn;
  nn->right = copyPreOrder(root->right);
  if (nn->right != NULL) nn->right->par = nn;
  return nn;
}

// print an expression tree with parentheses and infix math notation
void LinkedBinaryTree::printExpression(Node* v) {
  // if node is external print the value
  if (v->left == NULL && v->right == NULL) { 
    cout << v->elt;
  }
  else {
    // if node is one operand operator, print the operator then left node
    if (arity(v->elt) == 1) { 
      cout << v->elt << "(";
      printExpression(v->left);
      cout << ")";
    }
    // for two operand operators, print both child nodes seperated by the operator
    else {
      cout << "(";
      printExpression(v->left);
      cout << " " << v->elt << " ";
      printExpression(v->right);
      cout << ")";
    }
  }
}

// helps evaluate the expression tree, evaluates one operation
double evalOp(string op, double x, double y = 0) {
  double result;
  if (op == "+")
    result = x + y;
  else if (op == "-")
    result = x - y;
  else if (op == "*")
    result = x * y;
  else if (op == "/") {
    result = x / y;
  } else if (op == ">") {
    result = x > y ? 1 : -1;
  } else if (op == "abs") {
    result = abs(x);
  } else
    result = 0;
  return isnan(result) || !isfinite(result) ? 0 : result;
}

// evaluate the expression tree
double LinkedBinaryTree::evaluateExpression(const Position& p, double a,
                                            double b) {
  if (!p.isExternal()) {
    auto x = evaluateExpression(p.left(), a, b);
    if (arity(p.v->elt) > 1) {
      auto y = evaluateExpression(p.right(), a, b);
      return evalOp(p.v->elt, x, y);
    } else {
      return evalOp(p.v->elt, x);
    }
  } else {
    if (p.v->elt == "a")
      return a;
    else if (p.v->elt == "b")
      return b;
    else
      return stod(p.v->elt);
  }
}

// delete a subtree starting at and including node del
void LinkedBinaryTree::deleteSubtree(Node* del) {
  if(del != _root) { // remove the pointer to the node which will be deleted
    auto parent = del->par;
    (parent->left == del) ? parent->left = NULL : parent->right = NULL;
  }
  else { // remove the pointer to the node which will be deleted
    _root = NULL;
  }
  // delete the subtree rooted by node del
  PositionList delpl;
  preorder(del, delpl);
  for (auto& p : delpl) delete p.v;
}

// pick a random subtree to delete
void LinkedBinaryTree::deleteSubtreeMutator(mt19937& rng) {
  PositionList pl = positions();
  Position del = pl[randInt(rng, 0, pl.size()-1)];
  deleteSubtree(del.v);
}

// add a randomly generated subtree to the location of the missing leaf (just deleted by deleteSubtreeMutator)
void LinkedBinaryTree::addSubtreeMutator(mt19937& rng, const int maxDepth) {
  if (!empty()) { // find the missing leaf and add a random subtree
    PositionList pl = positions();
    Position parent;
    for (auto& p : pl) {
      if (isOp(p.v->elt) && (p.v->left == NULL || (arity(p.v->elt) == 2 && p.v->right == NULL))) {
        parent = p;
        break;
      }
    }
    LinkedBinaryTree subtree;
    subtree.addRoot(randElem(rng));
    subtree.randomExpressionTree(maxDepth - parent.v->depth() - 2, rng);
    (parent.v->left == NULL) ? addLeftChild(parent, subtree.root()) : addRightChild(parent, subtree.root());
  }
  else { // missing leaf is the root so generate a new random tree
    addRoot(randOperator(rng)); // generate a operator root as we know the problem isn't simple enough to solve with one parameter/constant
    randomExpressionTree(maxDepth, rng);
  }
}

// < operator overload for LinkedBinaryTree class
bool operator<(const LinkedBinaryTree& x, const LinkedBinaryTree& y) {
  return x.getScore() < y.getScore();
}

// creates an espression tree from a postfix string
LinkedBinaryTree createExpressionTree(string postfix) {
  stack<LinkedBinaryTree> tree_stack;
  stringstream ss(postfix);
  // Split each line into words
  string token;
  while (getline(ss, token, ' ')) {
    LinkedBinaryTree t;
    if (!isOp(token)) {
      t.addRoot(token);
      tree_stack.push(t);
    } else {
      t.addRoot(token);
      if (arity(token) > 1) {
        LinkedBinaryTree r = tree_stack.top();
        tree_stack.pop();
        t.addRightChild(t.root(), r.root());
      }
      LinkedBinaryTree l = tree_stack.top();
      tree_stack.pop();
      t.addLeftChild(t.root(), l.root());
      tree_stack.push(t);
    }
  }
  return tree_stack.top();
}

// recursively creates a random expression tree
void LinkedBinaryTree::randomExpressionTree(const Position& p, const int& maxDepth, mt19937& rng) {
  if (isOp(p.v->elt)) { 
    if (depth() < maxDepth - 1) { // if it will have children and hasn't reached maxDepth generate child nodes
      Node* x = new Node;
      x->elt = randElem(rng);
      addLeftChild(p,x);
      if (isOp(*p.left())) randomExpressionTree(p.left(), maxDepth, rng);
      if (arity(p.v->elt) > 1) { // operators with two operands
        x->elt = randElem(rng);
        addRightChild(p,x);
        if (isOp(*p.right())) randomExpressionTree(p.right(), maxDepth, rng);
      }
      delete x;
    }
    else { // max depth almost reached and final leafs are added
      Node* x = new Node;
      x->elt = randOperand(rng); // ensure leaf isn't an operator
      addLeftChild(p,x); 
      if (arity(p.v->elt) > 1) { // operators with two operands
        x->elt = randOperand(rng);
        addRightChild(p,x);
      }
      delete x;
    }
  }
}

// returns a randomly created expression tree
LinkedBinaryTree createRandExpressionTree(int max_depth, mt19937& rng) {
  LinkedBinaryTree t;
  t.addRoot(randOperator(rng)); // generate a operator root as we know the problem isn't simple enough to solve with one parameter/constant
  t.randomExpressionTree(max_depth, rng);
  return t;
}

// evaluate tree t in the cart centering task
void evaluate(mt19937& rng, LinkedBinaryTree& t, const int& num_episode,
              bool animate) {
  cartCentering env;
  double mean_score = 0.0;
  double mean_steps = 0.0;
  for (int i = 0; i < num_episode; i++) {
    double episode_score = 0.0;
    int episode_steps = 0;
    env.reset(rng);
    while (!env.terminal()) {
      int action = t.evaluateExpression(env.getCartXPos(), env.getCartXVel());
      episode_score += env.update(action, animate);
      episode_steps++;
    }
    mean_score += episode_score;
    mean_steps += episode_steps;
  }
  t.setScore(mean_score / num_episode);
  t.setSteps(mean_steps / num_episode);
}

// comparator for lexicographic comparison of two trees
class LexLessThan {
  public:
    bool operator()(const LinkedBinaryTree& t1, const LinkedBinaryTree& t2) const {
      if (abs(t1.getScore()-t2.getScore()) < 0.01) {
        return t1.size() > t2.size(); // favours simpler trees
      }
      else {
        return t1.getScore() < t2.getScore();
      }
    }
};

// crosses/swaps a random subtree from each tree
void LinkedBinaryTree::crossover(LinkedBinaryTree& otherTree, mt19937& rng) {
  // find the first subtree, save to a temp location, and delete from original tree
  PositionList pl1 = positions();
  Node* swapNode1 = pl1[randInt(rng, 1, pl1.size()-1)].v;
  LinkedBinaryTree subtree1;
  subtree1.addRoot(swapNode1);
  Node* parNode1 = swapNode1->par;
  deleteSubtree(swapNode1);
  // find the second subtree and use it to replace the original subtree on tree 1
  PositionList pl2 = otherTree.positions();
  Node* swapNode2 = pl2[randInt(rng, 1, pl2.size()-1)].v;
  (parNode1->left == NULL) ? addLeftChild(Position(parNode1), swapNode2) : addRightChild(Position(parNode1), swapNode2);
  // delete the second subtree from tree 2 and replace it with the first subtree from the temp location
  Node* parNode2 = swapNode2->par;
  deleteSubtree(swapNode2);
  (parNode2->left == NULL) ? addLeftChild(Position(parNode2), subtree1.root()) : addRightChild(Position(parNode2), subtree1.root());
}

int main() {
  mt19937 rng(42);
  // Experiment parameters
  const int NUM_TREE = 50;
  const int MAX_DEPTH_INITIAL = 1;
  const int MAX_DEPTH = 20;
  const int NUM_EPISODE = 20;
  const int MAX_GENERATIONS = 100;
  const float CROSSOVER_RATE = 0.5; // faction of trees which will be crossed each generation

  // Create an initial "population" of expression trees
  vector<LinkedBinaryTree> trees;
  for (int i = 0; i < NUM_TREE; i++) {
    LinkedBinaryTree t = createRandExpressionTree(MAX_DEPTH_INITIAL, rng);
    trees.push_back(t);
  }

  // Genetic Algorithm loop
  LinkedBinaryTree best_tree;
  std::cout << "generation,fitness,steps,size,depth" << std::endl;
  for (int g = 1; g <= MAX_GENERATIONS; g++) {

    // Fitness evaluation
    for (auto& t : trees) {
      if (t.getGeneration() < g - 1) continue;  // skip if not new
      evaluate(rng, t, NUM_EPISODE, false);
    }

    // // sort trees using overloaded "<" op (worst->best)
    // std::sort(trees.begin(), trees.end());

    // sort trees using comparaor class (worst->best)
    std::sort(trees.begin(), trees.end(), LexLessThan());

    // erase worst 50% of trees (first half of vector)
    trees.erase(trees.begin(), trees.begin() + NUM_TREE / 2);

    // Print stats for best tree
    best_tree = trees[trees.size() - 1];
    std::cout << g << ",";
    std::cout << best_tree.getScore() << ",";
    std::cout << best_tree.getSteps() << ",";
    std::cout << best_tree.size() << ",";
    std::cout << best_tree.depth() << std::endl;

    // Selection and mutation
    while (trees.size() < NUM_TREE) {
      // Selected random "parent" tree from survivors
      LinkedBinaryTree parent = trees[randInt(rng, 0, (NUM_TREE / 2) - 1)];
      
      // Create child tree with copy constructor
      LinkedBinaryTree child(parent);
      child.setGeneration(g);
      
      // Mutation
      // Delete a randomly selected part of the child's tree
      child.deleteSubtreeMutator(rng);
    
      // Add a random subtree to the child
      child.addSubtreeMutator(rng, MAX_DEPTH);
      
      trees.push_back(child);
    }

    // Crossover
    // Randomly choose two different trees and swap random subtrees from each
    for (int i = 0; i < CROSSOVER_RATE*NUM_TREE/2; i++) {
      LinkedBinaryTree t1 = trees[randInt(rng, 0, NUM_TREE - 1)];
      LinkedBinaryTree t2;
      do {  
        t2 = trees[randInt(rng, 0, NUM_TREE - 1)];
      } while (t1.root() == t2.root());
      t1.crossover(t2, rng);
    }
  }

  // // Evaluate best tree with animation
  const int num_episode = 3;
  evaluate(rng, best_tree, num_episode, true);

  // Print best tree info
  std::cout << std::endl << "Best tree:" << std::endl;
  best_tree.printExpression();
  std::cout << endl;
  std::cout << "Generation: " << best_tree.getGeneration() << endl;
  std::cout << "Size: " << best_tree.size() << std::endl;
  std::cout << "Depth: " << best_tree.depth() << std::endl;
  std::cout << "Fitness: " << best_tree.getScore() << std::endl << std::endl;
}