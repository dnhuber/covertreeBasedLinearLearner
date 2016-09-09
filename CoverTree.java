package covertree;

/**
 * This class provides a Java version of the cover tree nearest neighbor algorithm.
 * It is based on Thomas Kollar's version of "Cover Trees for Nearest Neighbor" by 
 * Langford, Kakade, Beygelzimer (2007). The original algorithm is extended towards
 * selecting K centers which are maximally different from one another from an online sample.
 * 
 * Date of creation: 2013-02-08
 * Copyright (c) 2013, Nils Loehndorf
 * 
 * The software is provided 'as-is', without any express or implied
 * warranty. In no event will the author be held liable for any damages
 * arising from the use of this software. Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it freely.
 * 
 * @author Nils Loehndorf
 *
 * The functionality is extended to predict linear values with a Cover Tree based Linear Learner.
 * The first column in the csv has to be the independent variable, the other the dependent variables.
 * This extension was done as part of a bachelor thesis supervised by Nils Loehndorf.
 * @author Daniel Huber
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

public class CoverTree<E> {

	int maxLevel;
	int minLevel;
	double base;
	int maxMinLevel;
	Node<E> rootNode;
	boolean hasBounds;
	double[] min;
	double[] max;
	int[] numLevels;
	int maxNumLevels = 500;
	int minNumLevels = -500;

	/**
	 * Create a cover tree at level zero which automatically expands above and below.
	 */
	public CoverTree () {
		this.maxMinLevel = Integer.MIN_VALUE;
		this.numLevels = new int[maxNumLevels-minNumLevels];
		this.base = 1.2;
	}

	/**
	 * Create a cover tree which stops increasing the minimumLevel as soon as the given number of nodes is reached.
	 */
	public CoverTree (double base, int maxMinLevel) {
		this.base = base;
		this.maxMinLevel = maxMinLevel;
		if (maxMinLevel>0) {
			this.maxLevel = maxMinLevel;
			this.minLevel = maxMinLevel;
		}
		this.numLevels = new int[maxNumLevels-minNumLevels];
	}

	/**
	 * Set the minimum levels of the cover tree by defining the maximum exponent of the base (default = 500).
	 */
	public void setMaxNumLevels (int max) {
		maxNumLevels = max;
	}

	/**
	 * Set the minimum levels of the cover tree by defining the minimum exponent of the base (default = -500).
	 */
	public void setMinNumLevels (int min) {
		minNumLevels = min;
	}

	/**
	 * Points outside of the bounding box, will not be included. This allows for easy truncation.
	 * @param min
	 * @param max
	 */
	public void setBounds(double[] min, double[] max) {
		hasBounds = true;
		this.min = min;
		this.max = max;
	}

	/**
	 * Returns the maximum level of this tree.
	 * @return
	 */
	public int maxLevel() {
		return maxLevel;
	}

	/**
	 * Returns the minimum level of this tree.
	 * @return
	 */
	public int minLevel() {
		return minLevel;
	}

	void incNodes(int level) {
		numLevels[level-minNumLevels]++;
	}

	void decNodes(int level) {
		numLevels[level-minNumLevels]--;
	}

	/**
	 * Returns the size of the cover tree up to the given level (inclusive)
	 * @param level
	 * @return
	 */
	public int size(int level) {
		int sum = 0;
		for (int i=maxLevel; i>=level; i--)
			sum += numLevels[i-minNumLevels];
		return sum;
	}

	/**
	 * Returns the size of the cover tree
	 * @return
	 */
	public int size() {
		return size(minLevel);
	}

	void insertAtRoot(E element, double[] point) {
		//inserts the point above the root by successively increasing the cover of the root node until it
		//contains the new point, the old root is added as child of the new root
		Node<E> oldRoot = rootNode;
		double dist = oldRoot.distance(point);
		while (dist > Math.pow(base,maxLevel)) {
			Node<E> newRoot = new Node<E>(null,rootNode.element,rootNode.point);
			rootNode.setParent(newRoot);
			newRoot.addChild(rootNode);
			rootNode = newRoot;
			decNodes(maxLevel);
			incNodes(++maxLevel);
		}
		Node<E> newNode = new Node<E>(rootNode,element,point);
		rootNode.addChild(newNode);
		incNodes(maxLevel-1);
	}

	/**
	 * Insert a point into the tree. If the tree size is greater than k the lowest cover will be removed as long as it does not decrease tree size below k.
	 * @param point
	 */
	public boolean insert(E element, double[] point, int k) {
		boolean inserted = insert(element,point);
		//only do this if there are more than two levels
		if (maxLevel-minLevel>2) {
			//remove lowest cover if the cover before has a sufficient number of nodes
			if (size(minLevel+1)>=k) {
				removeLowestCover();
				//do not accept new nodes at the minimum level
				maxMinLevel = minLevel+1;
			}
			//remove redundant nodes from the minimum level
			if (size(minLevel)>=2*k) {
				removeNodes(k);
			}
		}
		return inserted;
	}

	/**
	 * Insert a point into the tree.
	 * @param point
	 */
	public boolean insert(E element, double[] point) {
		if (hasBounds) {
			//points outside of the bounding box will not be added to the tree
			for (int d=0; d<point.length; d++) {
				if (point[d]>max[d])
					return false;
				if (point[d]<min[d])
					return false;
			}
		}
		//if this is the first node make it the root node
		if (rootNode == null) {
			rootNode = new Node<E>(null,element,point);
			incNodes(maxLevel);
			return true;
		}
		//do not add if the new node is identical to the root node
		rootNode.distance = rootNode.distance(point);
		if (rootNode.distance == 0.)
			return false;
		//if the node lies outside the cover of the root node and its decendants then insert the node above the root node
		if (rootNode.distance > Math.pow(base,maxLevel+1)) {
			insertAtRoot(element,point);
			return true;
		}
		//usually insertion begins here
		List<Node<E>> coverset = new LinkedList<Node<E>>();
		//the initial coverset contains only the root node
		coverset.add(rootNode);
		int level = maxLevel;
		Node<E> parent = null; //the root node does not have a parent
		int parentLevel = maxLevel;
		while (true) {
			boolean parentFound = true;
			List<Node<E>> candidates = new LinkedList<Node<E>>();
			for (Node<E> n1 : coverset) {
				for (Node<E> n2 : n1.getChildren()) {
					if (n1.point!=n2.point) {
						//do not compute distance twice
						n2.distance = n2.distance(point) ;
						//do not add if node is already contained in the tree
						if (n2.distance == 0.)
							return false;
					}
					else 
						n2.distance = n1.distance;
					if (n2.distance < Math.pow(base,level)) {
						candidates.add(n2);
						parentFound = false;
					}
				}
			}
			//if the children of the coverset are further away the 2^level then an element of the
			//coverset is the parent of the new node
			if (parentFound)
				break;
			//select one node of the coverset as the parent of the node
			for (Node<E> n : coverset) {
				if (n.distance < Math.pow(base,level)) {
					parent = n;
					parentLevel = level;
					break;
				}
			}
			//set all nodes as the new coverset
			level--;
			coverset = candidates;
		}
		//if the point is a sibling of the root node, then the cover of the root node is increased
		if (parent == null) {
			insertAtRoot(element,point);
			return true;
		}
		if (parentLevel-1 < minLevel) {
			//if the maximum size is reached and this would only increase the depth of the tree then stop
			if (parentLevel-1 < maxMinLevel)
				return false;
			minLevel = parentLevel-1;
		}
		//otherwise add child to the tree
		Node<E> newNode = new Node<E>(parent,element,point);
		parent.addChild(newNode);
		//record distance to parent node and add to the sorted set of nodes where distance is used for sorting (needed for removal)
		incNodes(parentLevel-1);
		return true;
	}



	/**
	 * Removes the the cover at the lowest level of the tree.
	 */
	void removeLowestCover() {
		List<Node<E>> coverset = new LinkedList<Node<E>>();
		coverset.add(rootNode);
		int k = maxLevel;
		while(k-- > minLevel+1){
			List<Node<E>> nextCoverset = new LinkedList<Node<E>>();
			for (Node<E> n : coverset) 
				nextCoverset.addAll(n.getChildren());
			coverset = nextCoverset;
		}
		for (Node<E> n : coverset) 
			n.removeChildren();

		minLevel++;
	}


	/**
	 * Removes all but k points.
	 */
	List<Node<E>> removeNodes(int numCenters) {
		List<Node<E>> coverset = new LinkedList<Node<E>>();
		coverset.add(rootNode);
		int k = maxLevel;
		while(k-- > minLevel+1){
			List<Node<E>> nextCoverset = new LinkedList<Node<E>>();
			for (Node<E> n : coverset) 
				nextCoverset.addAll(n.getChildren());
			coverset = nextCoverset;
		}
		int missing = numCenters-coverset.size();
		if (missing < 0)
			System.err.println("Error: negative missing="+missing+" in coverset");
		//sucessively pick the node with the largest distance to the coverset and add it to the coverset
		LinkedList<Node<E>> candidates = new LinkedList<Node<E>>();
		for (Node<E> n : coverset) 
			for (Node<E> n2 : n.getChildren())
				if (n.point!=n2.point)
					candidates.add(n2);
		//only add candidates when the coverset is yet smaller then the number of desired centers
		if (coverset.size()<numCenters) {
			//compute the distance of all candidates to their parents and uncles
			for (Node<E> n1 : candidates) {
				double minDist = Double.POSITIVE_INFINITY;
				for (Node<E> n2 : n1.getParent().getParent().getChildren()) {
					double dist = n1.distance(n2.point);
					if (dist < minDist)
						minDist = dist;
				}
				n1.distance = minDist;
				if (minDist==Double.POSITIVE_INFINITY)
					System.err.println("Error: Infinite distance in k centers computation.");
			}
			do {
				Collections.sort(candidates);
				Node<E> newNode = candidates.removeLast();
				coverset.add(newNode);
				//update the distance of all candidates in the neighborhood of the new node
				for (Node<E> n : newNode.getParent().getParent().getChildren()) {
					if (n!=newNode) {
						double dist = newNode.distance(n.point);
						if (dist < newNode.distance)
							newNode.distance = dist;
					}
				}
			} while (coverset.size()<numCenters);
		}
		//finally remove all nodes that have not been selected from the tree to avoid confusing the nearest neighbor computation
		for (Node<E> n : candidates) {
			n.getParent().removeChild(n);
			decNodes(minLevel);
		}
		return coverset;
	}

	/**
	 * Retrieve the elemnet from the tree that is nearest to the given point with respect to the Euclidian distance.
	 * @param point
	 * @return
	 */
	public E getNearest(double[] point) {
		List<Node<E>> candidates = new LinkedList<Node<E>>();
		candidates.add(rootNode);
		rootNode.distance = rootNode.distance(point);
		double minDist = rootNode.distance;
		for (int i=maxLevel; i>minLevel; i--) {
			List<Node<E>> newCandidates = new LinkedList<Node<E>>();
			for (Node<E> n : candidates) {
				for (Node<E> n2 : n.getChildren()) {
					//do not compute distances twice
					if (n.point!=n2.point) {
						n2.distance = n2.distance(point);
						//minimum distance can be recorded here
						if (n2.distance<minDist)
							minDist = n2.distance;
					}
					else
						n2.distance = n.distance;
					newCandidates.add(n2);
				}
			}
			candidates.clear();
			//create a set of candidate nearest neighbors
			for (Node<E> n : newCandidates)
				if (n.distance < minDist + Math.pow(base,i))
					candidates.add(n);
		}
		for (Node<E> n : candidates) {
			if (n.distance == minDist) 
				return n.element;
		}
		return null;
	}

	/**
	 * Get the cover of the given level. All points at this level are guaranteed to be 2^i apart from one another.
	 * @param level
	 * @return
	 */
	public List<E> getCover(int level) {
		List<Node<E>> coverset = new LinkedList<Node<E>>();
		coverset.add(rootNode);
		int k = maxLevel;
		while(k-- > level){
			List<Node<E>> nextCoverset = new LinkedList<Node<E>>();
			for (Node<E> n : coverset) 
				nextCoverset.addAll(n.getChildren());
			coverset = nextCoverset;
		}
		List<E> cover = new LinkedList<E>();
		for (Node<E> n: coverset) {
			cover.add(n.element);
		}

		return cover;
	}

	/**
	 * Convert the Set<Node<E>> you get from getAllChildren() to a flat data array
	 * @author danielhuber
	 * @param set
	 * @param obs
	 * @param vars
	 * @return
	 */
	public double[] setToArray(Set<Node<E>> set, int obs, int vars){
		double[] data = new double[(vars+1)*obs];
		int arrayCounter=0;
		for(Node<E> node: set){
			for(int i=0;i<=vars;i++){
				data[arrayCounter]=node.point[i];
				arrayCounter++;
			}}
		return data;
	}

	
	/**
	 * do linear regression over coverset with apache ols regression
	 * @author danielhuber
	 * @param node
	 */
	public void doLinRegression(Node<E> node){

		boolean isSingular=false; //if matrix is Singular make true
		OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
		Set<Node<E>> children = sequpointFastChildren(node);
		int obs = children.size();
		int vars = rootNode.point.length-1; // the independet variables and one dependent variable
		double[] beta = null;
		double[] a=setToArray(children,obs,vars);
		try {
			ols.newSampleData( a,obs,vars); 
		}
		catch(IllegalArgumentException e) {
			System.out.print("Can't sample data: ");
			e.printStackTrace();
		}

		try{
			beta = ols.estimateRegressionParameters(); 
		}
		//a singular Matrix can not be inverted which is necessary for this linear regression algorithm.
		//it has not a single solution. could be managed with a different linear regression algorithm

		catch(SingularMatrixException e){
			isSingular=true;
		}
		catch(IllegalArgumentException e) { 
			System.out.print("Can't estimate parameters: ");
			e.printStackTrace();
		}

		if(!isSingular){
			node.myResidualSumsSquared= ols.calculateResidualSumOfSquares();
			node.myRSquared =ols.calculateRSquared();
			node.myBeta=beta;
		}
		else{
			//make sure beta has the right length is matrix is singular
			beta= new double[vars+1];
		}
		node.size=obs;


		//do Regression of siblingset
		Set<Node<E>> siblings=new HashSet<Node<E>>();
		if(!(node==rootNode)){		//root has no siblings
			siblings = sequpointFastChildren(node.parent);// get set of points in parent
			siblings.removeAll(children);	
		}//get set of points in parent that are not in children
		if( siblings.size()>rootNode.point.length){		//no regression possible if there are less values in siblings than features
			try{

			obs = siblings.size();
			ols.newSampleData(setToArray(siblings, obs, vars),obs,vars);
			beta = ols.estimateRegressionParameters();

			node.siblingResidualSumsSquared=ols.calculateResidualSumOfSquares();
			node.siblingRSquared = ols.calculateRSquared();
			node.siblingBeta = beta;
			node.siblingSize=obs;
			}
			
			//values set to zero since it will affect all nodes below this node that get compared.
			catch(SingularMatrixException e){
				node.siblingResidualSumsSquared=0;
				node.siblingRSquared = 0;
				double[] helper = new double [beta.length];
				for(double i:helper){
					i=0;
				}
				node.siblingBeta=helper;
				node.siblingSize=siblings.size();
			}

		}
		else{
			node.siblingResidualSumsSquared=0;
			node.siblingRSquared = 0;
			double[] helper = new double [beta.length];
			for(double i:helper){
				i=0;
			}
			node.siblingBeta=helper;
			node.siblingSize=siblings.size();
		}

		//if Matrix is Singular get worst AIC so that a upper node is chosen
		if(isSingular){
			node.AIC= Double.MAX_VALUE;
		}
		else{
			node.AIC= calcAIC(node);
		}

		node.setHasLinRegTrue();

	}



	/**
	 * calculate a altered version of AIC. every step in the model is seen as another partial model with additional parameters
	 * @author danielhuber
	 **/
	public double calcAIC(Node<E> node){

		int numOfModels=1;
		double setSize;
		if(rootNode.hasLinReg){
			setSize=rootNode.size;
		}
		else{
			setSize=sequpointFastChildren(rootNode).size();
		}
		
		//get highest representation of this collection of points to get the best siblingResidualSumsSquared

		Node<E> helper;




		double totalRSS = node.myResidualSumsSquared+node.siblingResidualSumsSquared;		//RSS of all parts of the Model summed up


		while(node != rootNode){	//do until all the RSS of all points in the Set are represented.
			helper=node;
			node=node.parent;
			if(!node.hasLinReg){
				doLinRegression(node);
			}
			totalRSS += node.siblingResidualSumsSquared; //add the RSS of the Regression over all Siblings to the totalRSS

			//only increase penalty when points represented by node != points represented by parent node
			if((node.size!=helper.size)){


				numOfModels++;
			}
			
		}

		return (setSize*Math.log(totalRSS/setSize)+(2*((rootNode.point.length)*numOfModels)));


	}

	/**
	 * do forecast for the input point.
	 * precondition: point.length is root.length-1
	 * @author danielhuber
	 * @param point
	 * @return
	 */
	public double getForecastAllAIC(double[] point){

		double[] beta = nearRegAllAIC(point).myBeta;

		double prediction=beta[0];
		for(int i=1; i<beta.length;i++){
			prediction+=beta[i]*point[i-1];
		}

		return prediction;
	}


	/**
	 * find best regression node with AIC
	 * @author danielhuber
	 * @param point
	 * @return
	 */
	public Node<E> nearRegAllAIC(double[] point){
		Node<E> nearest= getNearestNode(point);

		//go up the tree until the sample represented by the node is bigger than the number of features (variables)
		while(sequpointFastChildren(nearest).size()<rootNode.point.length){
			nearest=nearest.parent;
		}
		
		if(!nearest.hasLinReg){
			doLinRegression(nearest);
		}
		
		Node<E> best= nearest;
		
		
		
		while(!(nearest.equals(rootNode))){
			
			nearest=nearest.parent;

			if(!nearest.hasLinReg){ //check if hasLinReg is true
				if(sequpointFastChildren(nearest).size()<rootNode.point.length){
					continue;
				}
				doLinRegression(nearest);
			}

			
			//System.out.println("AIC: "+nearest.AIC);
			
			if(nearest.AIC<=best.AIC){
				best=nearest;
			}
		}
		if(rootNode.AIC<=best.AIC){
			best=rootNode;
		}

		//System.out.println("Point above: "+point[0]+"best AIC is: "+best.AIC);
		return best;
	}
	

	public Node<E> getNearestNode(double[] point){
		List<Node<E>> candidates = new LinkedList<Node<E>>();
		candidates.add(rootNode);
		rootNode.distance = rootNode.forecastDistance(point);
		//at start min distance is distance of point to root node.
		double minDist = rootNode.distance;
		for (int i=maxLevel; i>minLevel; i--) {
			List<Node<E>> newCandidates = new LinkedList<Node<E>>();
			for (Node<E> n : candidates) {
				for (Node<E> n2 : n.getChildren()) {
					//do not compute distances twice
					if (n.point!=n2.point) {
						n2.distance = n2.forecastDistance(point);
						//minimum distance can be recorded here
						if (n2.distance<minDist)
							minDist = n2.distance;

					}
					else{
						n2.distance = n.distance;

					}
					newCandidates.add(n2);

				}
			}
			candidates.clear();
			//create a set of candidate nearest neighbors
			for (Node<E> n : newCandidates)
				if (n.distance < minDist + Math.pow(base,i))
					candidates.add(n);
		}
		for (Node<E> n : candidates) {
			if (n.distance == minDist) 
				return n;
		}
		return null;

	}

	//get points that are represented by that node in a ConcurrentHashSet not parallel
	public HashSet<Node<E>> sequpointFastChildren(Node<E> node){
		HashSet<Node<E>> children = new HashSet<Node<E>>();//Create ConcurrentHashSet (not defined explicitly in Java)
		helpsequPointFastChildren(node,children);
		return children;
	}

	private void helpsequPointFastChildren(Node<E> node, HashSet<Node<E>> children){
		if (node.children.isEmpty()){
			children.add(node);
		}
		else{		
			node.children.forEach(n->helpsequPointFastChildren(n,children));

		}
		return;
	}



	public int getLevel(Node<E> node){
		int i=0;
		while(!node.equals(rootNode)){
			i++;
			node=node.parent;
		}
		return maxLevel-i;
	}


	/**
	 * Gets at least k centers which are maximally apart from each other. All remaining centers are removed from the tree. This function only works as designed
	 * when the function insert(point,k) has been used before to add points to the tree. Otherwise, it will return the cover one level above the bottom most level of the tree.
	 * @param number of centers
	 * @return
	 */
	public List<E> getKCenters(int numCenters) {
		List<Node<E>> coverset = removeNodes(numCenters);
		//create cover
		List<E> cover = new LinkedList<E>();
		for (Node<E> n: coverset) {
			cover.add(n.element);
		}
		return cover;

	}

	static double distance(double[] d1, double[] d2) {
		double sumSq = 0.;
		for (int i=0; i<d1.length; i++) {
			double d = d1[i]-d2[i];
			sumSq += d*d;
		}
		return sumSq;
	}

	@SuppressWarnings("hiding")
	class Node<E> implements Comparable<Node<E>> {

		Node<E> parent; 
		E element;
		List<Node<E>> children;
		double[] point;
		double distance;
		
		//variables added for cover tree based linear learner by daniel huber
		double [] myBeta;
		double myResidualSumsSquared;
		double myRSquared;
		double [] siblingBeta;
		double siblingResidualSumsSquared;
		double siblingRSquared;
		double AIC;
		boolean hasLinReg = false;
		int size;
		int siblingSize;

		//use for a child
		Node (Node<E> parent, E element, double[] point) {
			this.parent = parent;
			this.children = new LinkedList<Node<E>>();
			this.element = element;
			this.point = point;
		}

		Node<E> getParent() {
			return parent;
		}

		void setParent(Node<E> node) {
			parent = node;
		}

		void setMyBeta(double[] beta){
			myBeta =beta;
		}

		double[] getMyBeta(){
			return myBeta;
		}

		void setMyResidualSumsSquared(double rss){
			myResidualSumsSquared = rss;
		}
		double getMyResidualSumsSquared(){
			return myResidualSumsSquared;
		}

		void setSiblingBeta(double[] beta){
			siblingBeta =beta;
		}

		double[] getSiblingBeta(){
			return siblingBeta;
		}

		void setSiblingResidualSumsSquared(double rss){
			myResidualSumsSquared = rss;
		}
		double getSiblingResidualSumsSquared(){
			return myResidualSumsSquared;
		}

		void setAIC(double aic){
			AIC = aic;
		}

		double getAIC(){
			return AIC;
		}

		void setHasLinRegTrue(){
			hasLinReg = true;
		}

		boolean getHasLinReg(){
			return hasLinReg;
		}


		void addChild(Node<E> node) {
			children.add(node);
		}


		List<Node<E>> getChildren() {
			if (children.isEmpty()) {
				Node<E> n = new Node<E>(this,this.element,this.point);
				addChild(n);
			}
			return children;
		}

		void removeChild(Node<E> n) {
			children.remove(n);
		}

		void removeChildren() {
			children.clear();
		}
		
		/**
		 * Distance for forecast: 1st value of tree data is y, which we forecast
		 * @author danielhuber
		 * @param point
		 * @return
		 */
		double forecastDistance(double[] point) {
			double sumSq = 0.;
			double d;
			for (int i=1; i<this.point.length; i++) {
				d = this.point[i]-point[(i-1)];
				sumSq += d*d;
			}
			return Math.sqrt(sumSq);

		}

		double distance(double[] point) {
			double sumSq = 0.;
			for (int i=0; i<point.length; i++) {
				double d = this.point[i]-point[i];
				sumSq += d*d;
			}
			sumSq = Math.sqrt(sumSq);
			return sumSq;
		}


		@Override
		public int compareTo(Node<E> o) {
			if (distance < o.distance)
				return -1;
			if (distance > o.distance)
				return 1;
			return 0;
		}




	}

}
