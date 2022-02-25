#!/usr/bin/perl

# package HMM.pm
# A Perl package for Hidden Markov Models

# Author: MH Seabolt
# Last Updated: 4-14-2019	

# Current implementation is primarily for discrete-time Markov chain processes (represented as directed graph adjacency matrices).

# CHANGELOG 4-14-2020 MHS:
#    -- Changed constructor to automatically include a BEGIN state and an END state, including updating the prior probability transitions for this
#    -- Updated the Forward, Backward, etc. algorithms to utilitze these states correctly.

# IN PROGRESS:
#    -- Add I/O functions to write out a model to a flat text file, and a parser to read an reconstruct a model from file.

package HMM;

require Exporter;
@ISA = qw(Exporter);
@EXPORT = qw(sum lg log10 pow round);			 #Import from other packages

use strict;
use warnings;
use List::Util qw( sum first shuffle );
use Scalar::Util;
use Storable qw(dclone);
use Data::Dumper;
use Carp;
our $AUTOLOAD;

# @INC libraries for my PC and Biolinux or HS custom classes 
use MarkovNode;

# ESSENTIAL CPAN module for Sets
use Set::Scalar;

################################################################################## 

# Class data and methods 
# Attributes 
{
	# _states and _adj are inherited from Markov parent class
	my %_attribute_properties = (
		_states						=> [ ],					# 1D array containing the list of all *latent* states in the model, where each state will be a MarkovNode object
		_adj						=> [ ],					# 2D matrix representing the transition probabilities of change of state of latent variable
		_edj						=> [ ],					# 2D matrix representing the probability of a given observation given the state of the latent variable
		_observation_states			=> [ ],					# 1D array containing the list of all *observed* states in the model, where each state will be a MarkovNode object
		_prior_probabilities		=> [ ],					# 1D array representing the prior probabilities of all states of the latent variable (transitions from BEGIN to the first state)
		_posterior_probabilities	=> [ ],					# 1D array representing the posterior probabilities of all states of the latent variable transitioning to the END state
		_name 						=> '',					# An optional name for this HMM (this is really just for I/O purposes)
	);
	
	# Global variable counter
	my $_count = 0;
	
	# Return a list of all attributes
	sub _all_attributes	{
		keys %_attribute_properties;
	}
	
	# Return the default value for a given attribute
    	sub _attribute_default 	{
     	my( $self, $attribute ) = @_;
        	$_attribute_properties{$attribute};
    	}
    
	# Manage the count of existing objects
	sub get_count	{
		$_count;
	}
	sub _incr_count	{
		++$_count;
	}
	sub _decr_count	{
		--$_count;
	}	
}


############################################################
#                       CONSTRUCTORS                       #
############################################################

# The contructor method
# Construct a new graph (my $hmm = HMM->new() );
# Returns a scalar reference to a new HMM model object
sub new				{
	my ( $class, %arg ) = @_;
	
	# Create the new object
	my $self = bless {}, $class;

	foreach my $attribute ( $self->_all_attributes() ) {
        	# E.g. attribute = "_name",  argument = "name"
        	my ($argument) = ( $attribute =~ /^_(.*)/ );
        	# If explicitly given
        	if (exists $arg{$argument}) 	{
            	$self->{$attribute} = $arg{$argument};
        	}
        	else	{
            	$self->{$attribute} = $self->_attribute_default($attribute);
        	}
   	}
   	
	# Construction code specific to this class
	######################################################################################
	
	# Initialize a BEGIN state and an END state
	$self->add_state( "BEGIN" );
	$self->add_state( "END"   );
	
   	# If attribute args are given for _states or _observation_states, initialize them
  	if ( scalar @{$self->{_states}} > 0 )	{
  		my @names = @{$self->{_states}};				# Assumed that we gave a list of names for the states in the constructor
   		$self->{_states} = [ ];
   		for (my $u=0; $u < scalar @names; $u++ )	{
   			$self->add_state( $names[$u], $self->{_adj}->[$u], $self->{_edj}->[$u], $self->{_prior_probabilities}->[$u], $self->{_posterior_probabilities}->[$u] );
   		}
  	}
   	if ( scalar @{$self->{_observation_states}} > 0 )	{
   		my @names = @{$self->{_observation_states}};				# Assumed that we gave a list of names for the states in the constructor
   		$self->{_observation_states} = [ ];
   		for (my $u=0; $u < scalar @names; $u++ )	{
   			my @emission_column;
   			for (my $v=0; $u < scalar @{$self->{_states}}; $u++ )	{
   				push @emission_column, $self->{_edj}->[$u]->[$v];	
   			}
   			$self->add_observation_state( $names[$u], \@emission_column );
   		}
   	}
	
	# Choose a name for the model if one wasnt given by the user
	if ( not $self->{_name} )	{
		my $x = int(rand(100));
		$self->{_name} = "HMM$x";
	}
	
	######################################################################################
   	   	
    $class->_incr_count();
	return $self;
}

# The clone method
# All attributes are copied from the calling object, unless specifically overriden
# Called from an existing object ( Syntax: $cloned_obj = $obj->clone(); )
sub clone	{
	my ( $caller, %arg ) = @_;
	# Extract the class name from the calling object
	my $class =ref($caller);
		
	# Create a new object
	my $self = bless {}, $class;
		
	foreach my $attribute ( $self->_all_attributes() )	{
		my ($argument) = ( $attribute =~ /^_(.*)/ );
			
		# If explicitly given
		if ( exists $arg{$argument} )	{
			$self->{$attribute} = $arg{$argument};
		}
			
		# Otherwise, copy attribute of new object from the calling object
		else	{
			$self->{$attribute} = $caller->{$attribute};
		}
	}
	$self->_incr_count();
	return $self;
}

#######################################
# Autoload getters and setters
sub AUTOLOAD {
    	my ( $self, $newvalue ) = @_;
    	my ( $operation, $attribute ) = ( $AUTOLOAD =~ /(get|set)(_\w+)$/ );
    
    	# Is this a legal method name?
    	unless( $operation && $attribute ) {
        	croak "Method name $AUTOLOAD is not in the recognized form (get|set)_attribute\n";
    	}
    	unless( exists $self->{$attribute} ) {
        	croak "No such attribute $attribute exists in the class ", ref($self);
    	}

    	# Turn off strict references to enable "magic" AUTOLOAD speedup
    	no strict 'refs';

    	# AUTOLOAD accessors
    	if( $operation eq 'get' ) 	{
        	# Install this accessor definition in the symbol table
        	*{$AUTOLOAD} = sub {
            	my ($self) = @_;
          	$self->{$attribute};
     	};
    }
    # AUTOLOAD mutators
    elsif( $operation eq 'set' ) 	{
		# Set the attribute value
        	$self->{$attribute} = $newvalue;

        	# Install this mutator definition in the symbol table
        	*{$AUTOLOAD} = sub {
        		my ($self, $newvalue) = @_;
            	$self->{$attribute} = $newvalue;
        	};
    }

    # Turn strict references back on
    use strict 'refs';

    # Return the attribute value
    return $self->{$attribute};
}

############################################################
#                BASIC UTILITY SUBROUTINES                 #
############################################################

# This hash is associated with the _stateToIndex() subroutine, but it must be outside the subroutine scope to be maintainable and accessible.
my $h = 0;			# The initial index in the prebuilt hash of states below, we will increment it if needed
my %HMMIndex = ();
my %HMMReverseIndex = ();		# Associated with the _indexToState() subroutine and automatically updated by _stateToIndex()
my %HMMNames = ();			
my %HMMNamesIndex = ();
my %HMMNamesState = ();

# Converts key current character into index
sub _stateToIndex	{
	my ( $self, $state )	= @_;
	return if ( not $state );		# Sanity check
	my $index;
	
	if ( exists $HMMIndex{$state}  ) 	{
		$index = $HMMIndex{$state};
	}
	# If the index doesnt exist in the Index hash, then add it at the end and increment the value
	# This should be fine for multiple letter k-mer style indices and some symbols
	else		{
		$HMMIndex{$state} = $h;				# Be careful here, as odd symbols may cause errors
		$HMMReverseIndex{$h} = $state;
		my $name = $state->{_name};
		$HMMNamesIndex{$h} = $name;
		$HMMNames{$name} = $h;
		$HMMNamesState{$name} = $state;
		$h++;
		$index = $HMMIndex{$state};
	}

	return $index;
}

# Converts key current index into character
sub _indexToState	{
	my ( $self, $index )	= @_;
	my $state;
	if ( exists( $HMMReverseIndex{$index} ) ) 	{
		$state = $HMMReverseIndex{$index};
	}
	# If the index doesnt exist in the ReverseIndex hash, then there is nothing we can do...
	
	return $state;
}

# Converts from an index to a state NAME
sub _indexToName	{
	my ( $self, $index ) = @_;
	my $name;
	if ( exists $HMMNamesIndex{$index} )	{
		$name = $HMMNamesIndex{$index};
	}
	return $name;
}

# Converts from the name of a state to the index
sub _nameToIndex	{
	my ( $self, $name ) = @_;
	my $index;
	if ( exists $HMMNames{$name} )	{
		$index = $HMMNames{$name};
	}
	return $index;
}

sub _nameToState	{
	my ( $self, $name ) = @_;
	my $state;
	if ( exists $HMMNames{$name} )	{
		$state = $HMMReverseIndex{ $HMMNames{$name} };
	}
	return $state;
}

################
# Observation hashs

my $o = 0;
my %ObsIndex = ();
my %ObsReverseIndex = ();
my %ObsNames = ();			
my %ObsNamesIndex = ();
my %ObsNamesState = ();

# Converts from the name of an observation to the index in @_observation_states
sub _obsStateToIndex	{
	my ( $self, $obs ) = @_;
	return if ( not $obs );		# Sanity check
	my $index;
	
	if ( exists $ObsIndex{$obs}  ) 	{
		$index = $ObsIndex{$obs};
	}
	# If the index doesnt exist in the Index hash, then add it at the end and increment the value
	# This should be fine for multiple letter k-mer style indices and some symbols
	else		{
		$ObsIndex{$obs} = $o;				# Be careful here, as odd symbols may cause errors
		$ObsReverseIndex{$o} = $obs;
		my $name = $obs->{_name};
		$ObsNamesIndex{$o} = $name;
		$ObsNames{$name} = $o;
		$ObsNamesState{$name} = $obs;
		$o++;
		$index = $ObsIndex{$obs};
	}

	return $index;
}

# Converts an index in @_observation_states to the name of the observation
sub _indexToObsState	{
	my ( $self, $index ) = @_;
	my $name;
	if ( exists $ObsReverseIndex{$index} )	{
		$name = $ObsReverseIndex{$index};
	}
	return $name;
}

# Converts from an index to an observation state NAME
sub _obsIndexToName	{
	my ( $self, $index ) = @_;
	my $name;
	if ( exists $ObsNamesIndex{$index} )	{
		$name = $ObsNamesIndex{$index};
	}
	return $name;
}

# Converts from the name of a state to the index
sub _obsNameToIndex	{
	my ( $self, $name ) = @_;
	my $index;
	if ( exists $ObsNames{$name} )	{
		$index = $ObsNames{$name};
	}
	return $index;
}

sub _obsNameToState	{
	my ( $self, $name ) = @_;
	my $state;
	if ( exists $ObsNames{$name} )	{
		$state = $ObsReverseIndex{ $ObsNames{$name} };
	}
	return $state;
}

#################################################################################

# Calculates the degree of a specified node
# Returns a scalar integer value of the degree of a node.
sub degree		{
	my ( $self, $u ) = @_;
	my $degree = 0;	
	my $r = $self->_nameToIndex( $u );	
	my @children = @{ $self->{_adj}->[$r] };
	
	for ( my $v = 0; $v < scalar @children; $v++ )	{
		next if ( $children[$v] == 0 || $children[$v] eq "inf" || $children[$v] eq "nan" );	# Increment degree if the edge exists
		$degree++;
		$degree++ if ( $self->_stateToName($self->{_states}->[$v]) ~~ $u );			# Increment degree AGAIN if we have a self loop.
	}
	return $degree;
}

# Calculates the outdegree of a specified node (directed edges originating from this node, to another node) --> the row in the Adj
# Returns a scalar integer value of the degree of a node.
# Only use for a directed graph structure
sub outdegree		{
	my ( $self, $u ) = @_;
	my $degree = 0;	
	my $r = $self->_nameToIndex( $u );	
	my @children = @{ $self->{_adj}->[$r] };
	
	for ( my $v = 0; $v < scalar @children; $v++ )	{
		$degree++ if ( $children[$v] != 0 || $children[$v] ne "inf" || $children[$v] ne "nan" );	# Increment degree if the edge exists

	}
	return $degree;
}

# Calculates the outdegree of a specified node (directed edges originating from this node, to another node) --> the column in the Adj
# Returns a scalar integer value of the degree of a node.
# Only use for a directed graph structure
sub indegree		{
	my ( $self, $u ) = @_;
	my $degree = 0;
	my @children;
	my $c = $self->_nameToIndex( $u );	
	foreach my $row ( @{ $self->{_adj}	} )	{
		push @children, $self->{_adj}->[$row]->[$c];
	}
	
	for ( my $v = 0; $v < scalar @children; $v++ )	{
		$degree++ if ( $children[$v] != 0 || $children[$v] ne "inf" || $children[$v] ne "nan" );	# Increment degree if the edge exists

	}
	return $degree;
}

# Brandes algorithm for betweeness-centrality
# Computes the shortest-path betweenness centrality for all nodes
# Betweenness centrality of a node V is the sum of the fraction of all-pairs shortest paths that pass through V
# NOTE: For weighted graphs the edge weights must be greater than zero.   Zero edge weights can produce an infinite number of equal length paths between pairs of nodes. 
# Returns a hash with KEYS = nodes, VALUES= betweenness centrality as the value
sub betweenness_centrality		{
	my ( $self ) = @_;
	my @names;
	
	# Initalize the hash of values with KEYS= vertices and VALUES= betweeness centrality value
	my %C = ();
	my %P = ();
	my %G = ();
	my %D = ();
	my %E = ();
	foreach my $state ( @{$self->{_states}} )	{
		my $name = $self->_stateToName($state);
		push @names, $name;
		$C{$name} = 0;
	}
	
	foreach my $s ( @names )	{
		# Initialize various structures for each node in the graph
		my @S;
		my @q;
		foreach ( @names )	{
			$P{$_} = [];
			$G{$_} = 0;
			$D{$_} = -1;
			$E{$_} = 0;
		}
		$G{$s} = 1;
		$D{$s} = 0;
		push @q, $s;
		
		# Loop
		while ( scalar @q > 0 )	{
			my $v = shift(@q);
			my $vname = $self->_nameToIndex($v);
			push @S, $v;
			foreach my $w ( @names )		{
				my $wname = $self->_nameToIndex($w);
				next if ( $self->{_adj}->[$v]->[$w] == 0 || $self->{_adj}->[$v]->[$w] eq "inf" || $self->{_adj}->[$v]->[$w] eq "nan" );
				if ( $D{$w} < 0 )	{
					push @q, $w;
					$D{$w} = $D{$v} + 1;
				}
				if ( $D{$w} == $D{$v} + 1 )	{
					$G{$w} = $G{$w} + $G{$v};
					push @{$P{$w}}, $v;
				}
			}
		}
		
		while ( scalar @S > 0 )	{
			my $w = pop @S;
			foreach my $v ( @{ $P{$w} } )	{
				$E{$v} = $E{$v} + ($G{$v}/$G{$w}) * (1+$E{$w});
			}
			if ( $w !~ $s )	{
				$C{$w} = $C{$w} + $E{$w};
			}
		}
	}
	return \%C;
}

# Utility subroutine to make a deep copy of a Markov object (similar to the clone() subroutine)
sub copy_model	{
	my ( $self ) = @_;
	my $clone = dclone $self;
	return $clone;
}

# Utility subroutine to return an EMPTY adjacency matrix with the same structure (states) as the input.
sub empty	{
	my ( $self ) = @_;
	my @Adj = @{$self->get_adj};
	for ( my $u=0; $u < scalar @{ $self->{_states} }; $u++ )	{
		my $state = $self->_indexToState( $u );	 
		
		# This loop just resizes the array and initializes zeros where needed
		for ( my $v=0; $v < scalar @{ $self->{_states} }; $v++ )	{
			my $child = $self->_indexToState( $v );	
			$child->{_children}->[$v] = $child if ( $Adj[$u]->[$v] && $Adj[$u]->[$v] > 0 );
			next if ( $Adj[$u]->[$v] );
			$Adj[$u]->[$v] = 0;			
		}	
	}
	$self->set_adj( \@Adj );
}


# When an object is no longer being used, garbage collect it and adjust count of existing objects
sub DESTROY	{
	my ( $self ) = @_;
	$self->_decr_count();
}

#################################################################################
#          INSERTION AND DELETION SUBROUTINES FOR TRANSITION MATRIX             #
#################################################################################

# Adds a vertex (a "state") to the Markov chain graph of the LATENT VARIABLE
# Updates the _states array and the %Index /%ReverseIndex hashes
# Transition probabilties should be a hash ref of probabilities with keys as the state objects and the values as the probability weights
# Operates DIRECTLY on the Markov object.
sub add_state		{
	my ( $self, $new_state, $transitions, $emissions, $prior, $post ) = @_;

	# Initialize a new node object
	$new_state = ( $new_state )? $new_state : $h;		# ALL states must have a name! Use $p as a placeholder if no name is supplied.
	my $node = MarkovNode->new( "name" => "$new_state" );
	
	# Update the requisite hashes and lists with the new node
	push @{ $self->{_states} }, $node;
	$self->_stateToIndex( $node );
	
	# Update the list of priors and set the transition from BEGIN to this node
	$prior = ( $prior && $prior >= 0 )? $prior : 0;
	push @{ $self->{_prior_probabilities} }, $prior;
	
	# Update the list of posteriors and set the transition from this node to END
	$post = ( $post && $post >= 0 )? $post : 0;
	push @{ $self->{_posterior_probabilities} }, $post;
	
	# Update _adj and _emission_adj, and set the children of the new state
	my @Adj = @{$self->get_adj};
	my @Edj = @{$self->get_edj};
	for ( my $u=0; $u < scalar @{ $self->{_states} }; $u++ )	{
		my $state = $self->_indexToState( $u );	 
		$Edj[$u] = [ ] unless ( $Edj[$u] );
		
		# This loop just resizes the array and initializes zeros where needed in _adj
		for ( my $v=0; $v < scalar @{ $self->{_states} }; $v++ )	{
			my $child = $self->_indexToState( $v );	
			$child->{_children}->[$v] = $child if ( $Adj[$u]->[$v] && $Adj[$u]->[$v] > 0 );
			next if ( $Adj[$u]->[$v] );
			$Adj[$u]->[$v] = 0;	
		}
		
		# Set the prior and posterior transitions in Adj and Edj
		$Adj[0]->[$u] = $prior;
		$Adj[1]->[$u] = $post;
		
		#$Edj[0]->[$u] = 0;		# The BEGIN state is mute, there are no emissions from it
		#$Edj[1]->[$u] = 0;		# The END state is also assumed to be mute, there are no emissions from it either.
	}
	
	# The same loop as above, for _edj, since _edj does not 
	# have to be the same size (nor even symmetrical NxN) as _adj
	# All this does here is add a new COLUMN to Edj for the new state (does NOT add a column!)
	for ( my $u=0; $u < scalar @{$self->{_observation_states}}; $u++ )	{
		for ( my $v=0; $v < scalar @{$self->{_states}}; $v++ )	{
			next if ( $Edj[$u]->[$v] );
			$Edj[$u]->[$v] = 0;	
			$Edj[$u]->[$v] = $emissions->[$u] if ( $emissions->[$u] );
		}
	}	
	
	# Populate the correct transition and emission probability data into the correct matrices, if we have that data
	if ( $transitions )	{		# by row
		for ( my $w=0; $w < scalar @{$transitions}; $w++ )	{
			my $state = $self->_indexToState( $w );
			$node->{_children}->[$w] = $state if ( $transitions->[$w] > 0 );
			$Adj[-1]->[$w] = $transitions->[$w];
		}
	}
	if ( $emissions && ref($emissions) eq 'ARRAY' )	{		# by column
		for ( my $w=0; $w < scalar @{$emissions}; $w++ )	{
			my $state = $self->_indexToState( $w );
			$state->{_observations}->[$w] = $self->_indexToObsState($w) if ( $emissions->[$w] > 0 );
			
		}
	}
	$self->set_adj( \@Adj );
	$self->set_edj( \@Edj );
}


# Deletes a state from both the list of states, adj, and edj
sub delete_state	{
	my ( $self, $name ) = @_;
	my $kill_state = $self->_nameToState( $name );
	
	# Update _states
	my $index = $self->_stateToIndex( $kill_state );
	
	# Update _adj and the children of the affected nodes
	for ( my $u=0; $u < scalar @{ $self->{_adj} }; $u++ )	{
		# Delete the column of the dead node from adj and edj
		splice( @{ $self->{_adj}->[$u] }, $index, 1 );
		my $state = $self->{_states}->[$u];
		splice( @{ $state->{_children} }, $index, 1 );
	}
	
	# Delete the column in _edj
	for ( my $u=0; $u < scalar @{ $self->{_observation_states} }; $u++ )	{		# Need to correct this code, we are currently modifying the data structure that we are looping over
		splice( @{ $self->{_edj}->[$u] }, $index, 1 );
	}
	
	# Finally, delete the entire row for the node in _adj, _priors, and _states
	splice( @{ $self->{_adj} }, $index, 1 );
	splice( @{ $self->{_states} }, $index, 1 );
	splice( @{ $self->{_prior_probabilities} }, $index, 1 );
	$kill_state->DESTROY;
	
	# Update the organizational hashes --> requires resetting and rehashing them all
	$h = 0;			
	%HMMIndex = ();
	%HMMReverseIndex = ();		
	%HMMNames = ();			
	%HMMNamesIndex = ();
	%HMMNamesState = ();
	for ( my $i=0; $i < scalar @{$self->{_states}}; $i++ )	{
		my $state = $self->{_states}->[$i];
		$self->_stateToIndex( $state );
	}
}

# Adds an edge with weight $weight (required) between source state $u and target state $v
# If $weight is not given as an arg, then nothing will happen
# If an edge already exists between $u --> $v, then this will overwrite it
sub add_edge		{
	my ( $self, $uname, $vname, $weight, $matrix ) = @_;
	my $u = $self->_nameToState( $uname );
	my $v = $self->_nameToState( $vname );
	
	# Which matrix are we adding an edge to?
	if ( $matrix =~ /^[Tt]/ )	{	$matrix = "transition";	}
	elsif ( $matrix =~ /^[Ee]/ )	{	$matrix = "emission";	}
	else						{	$matrix = "transition";	}		# Default to transition (_adj) matrix	
	
	# Sanity checks
	if ( not $u || not $v || not $weight )	{
		warn " --- HMM::add_edge() ERROR:  missing subroutine argument!\n";
		return;
	}
	elsif ( not exists $HMMNames{$u} || not exists $HMMNames{$v} )	{
		warn " --- HMM::add_edge() ERROR:  cannot add edges between nodes that don't exist!\n";
		return;
	}
	
	# Update the edge in the requested matrix, and then 
	# Set the updated row in the correct state node
	if ( $matrix eq "transition" )	{
		$self->{_adj}->[ $self->_stateToIndex($u) ]->[ $self->_stateToIndex($v) ] = $weight;
		$u->set_children( $self->{_adj}->[$u] );
	}
	if ( $matrix eq "emission" )	{
		$self->{_edj}->[ $self->_stateToIndex($u) ]->[ $self->_stateToIndex($v) ] = $weight;
		$u->set_observations( $self->{_edj}->[$u] );
	}
}

# Deletes an edge (here, we are actually just resetting the Pr to 0) between source $u and target $v.
sub delete_edge	{
	my ( $self, $uname, $vname, $matrix ) = @_;
	my $u = $self->_nameToState( $uname );
	my $v = $self->_nameToState( $vname );
	
	# Which matrix are we deleting an edge from?
	if ( $matrix =~ /^[Tt]/ )	{	$matrix = "transition";	}
	elsif ( $matrix =~ /^[Ee]/ )	{	$matrix = "emission";	}
	else						{	$matrix = "transition";	}		# Default to transition (_adj) matrix
	
	# Sanity checks
	if ( not exists $HMMIndex{$u} || not exists $HMMIndex{$v} )	{
		warn " --- HMM::delete_edge() ERROR:  cannot delete edges between nodes that don't exist!\n";
		return;
	}
	
	# Update the edge Pr to 0
	if ( $matrix eq "transition" )	{
		$self->{_adj}->[ _stateToIndex($u) ]->[ _stateToIndex($v) ] = 0;
	}
	if ( $matrix eq "emission" )	{
		$self->{_edj}->[ _stateToIndex($u) ]->[ _stateToIndex($v) ] = 0;
	}
	
	# Update the children of the $u (the only node with a new child
	my @children = @{ $self->{_adj}->[$u] };
	$u->set_children( \@children );
}

# Adds an edge with weight $weight (required) between source state $u and target state $v
# If $weight is not given as an arg, then nothing will happen
# If an edge already exists between $u --> $v, then this will overwrite it
# This code is the same as add_edge(), but functions as an alias.
sub update_edge		{
	my ( $self, $uname, $vname, $weight, $matrix ) = @_;
	my $u = $self->_nameToState( $uname );
	my $v = $self->_nameToState( $vname );
	
	# Which matrix are we adding an edge to?
	if ( $matrix =~ /^[Tt]/ )	{	$matrix = "transition";	}
	elsif ( $matrix =~ /^[Ee]/ )	{	$matrix = "emission";	}
	else						{	$matrix = "transition";	}		# Default to transition (_adj) matrix	
	
	# Sanity checks
	if ( not $u || not $v || not $weight )	{
		warn " --- HMM::update_edge() ERROR:  missing subroutine argument!\n";
		return;
	}
	elsif ( not exists $HMMIndex{$u} || not exists $HMMIndex{$v} )	{
		warn " --- HMM::update_edge() ERROR:  cannot alter edges between nodes that don't exist!\n";
		return;
	}
	
	# Update the edge in the requested matrix, and then 
	# Set the updated row in the correct state node
	if ( $matrix eq "transition" )	{
		$self->{_adj}->[ $self->_stateToIndex($u) ]->[ $self->_stateToIndex($v) ] = $weight;
		$u->set_children( $self->{_adj}->[$u] );
	}
	if ( $matrix eq "emission" )	{
		$self->{_edj}->[ $self->_stateToIndex($u) ]->[ $self->_stateToIndex($v) ] = $weight;
		$u->set_observations( $self->{_edj}->[$u] );
	}
}


# Alias for setting the prior probabilities for each/all states in the latent variable forming the Markov chain
# Accepts an array reference containing the new priors for all states, in the correct order
sub set_priors		{
	my ( $self, $priors ) = @_;
	
	# If no argument is given for $priors, then assume we want to set them as uniform/equal
	my @uniform_priors;
	if ( not $priors )	{
		my $k = 1 / scalar @{ $self->{_states} };
		for ( my $i=0; $i < scalar @{$self->{_states}}; $i++ )	{
			$uniform_priors[$i] = $k;
		}
		$priors = \@uniform_priors;
	}
	
	# Sanity check -- the length of the priors array should be equal to the number of states in the model
	# This additionally does not allow the user to pass an empty argument for priors (ie there is no default)
	if ( scalar @{$priors} != scalar @{$self->{_states}} )	{
		warn " --- HMM::set_priors() ERROR:  the number of priors given to the subroutine doesnt match the number of states in the HMM model!\n";
		return;
	}
	$self->{_prior_probabilities} = $priors;
}

# Set/update the prior probability of a given state
sub set_prior	{
	my ( $self, $name, $prior ) = @_;
	my $index = $self->_nameToIndex($name);
	$prior = ( $prior )? $prior : 0;		# Defaults to zero probability if you dont give it anything to set.
	$self->{_prior_probabilities}->[$index] = $prior;
}

# Appends a new observation state to the list of possible observations and update the appropriate hashes
# and initializes a new column in the Edj hash
sub add_observation_state	{
	my ( $self, $new_obs, $emissions ) = @_;
	
	# Initialize the new element in @observation_states and update the organizational hashes
	$new_obs = ( $new_obs )? $new_obs : $o;		
	my $node = MarkovNode->new( "name" => "$new_obs", "isObsState" => 1 );
	push @{ $self->{_observation_states} }, $node;
	my $index = $self->_obsStateToIndex( $node );
	
	# Append the new col onto _edj
	for ( my $u=0; $u < scalar @{$self->{_states}}; $u++ )	{
		$self->{_edj}->[$index]->[$u] = 0;			# Initialize to 0
		$self->{_edj}->[$index]->[$u] = $emissions->[$u] if ( $emissions );	
		
		# Update @observations for each latent state in the model
		my $latent_state = $self->_indexToState($u);
		for ( my $v=0; $v < scalar @{$self->{_observation_states}}; $v++ )	{
			my $observation_state = $self->_indexToObsState($v);
			next if $latent_state->{_observations}->[$v];
			$latent_state->{_observations}->[$v] = $observation_state;
		}
	}
}

# Deletes an observation state from the main Markov model, removes it from the organizational lookup hashes, 
# and finally, deletes it from the list of possible observation states for all nodes in the Markov chain.
sub delete_observation_state	{
	my ( $self, $kill_obs ) = @_;
	
	# Delete the state from the set of observations
	my $index = $self->_obsToIndex($kill_obs);
	splice( @{$self->{_observation_states}}, $index, 1 );
	
	# Delete the column in _edj
	for ( my $u=0; $u < scalar @{$self->{_edj}->[0]}; $u++ )	{
		splice( @{$self->{_edj}->[$u]}, $index, 1 );
	}
	
	# Update the organizational hashes --> requires resetting them and rehashing
	$o = 0;
	%ObsIndex = ();
	%ObsReverseIndex = ();
	my %ObsNames = ();			
	my %ObsNamesIndex = ();
	my %ObsNamesState = ();
	for ( my $i=0; $i < scalar @{$self->{_observation_states}}; $i++ )	{
		my $obs = $self->{_observation_states}->[$i];
		$self->_obsStateToIndex( $obs );
	}
	
	# Delete the observation for all nodes in the latent variable Markov chain
	for ( my $j=0; $j < scalar @{$self->{_states}}; $j++ )	{
		my $state = $self->{_states}->[$j];
		$state->set_observations( $self->{_edj}->[$j] );
	}
}



#################################################################################
#                            OTHER SUBROUTINES                                  #
#################################################################################

# Chooses a random state from the list of ALL possible states in the graph, based on the priors for each state
# Works by getting all the shuffling the array to get a random one.
sub random_state		{
	my ( $self ) = @_;
	my $random_state = $self->sample_from_priors( $self->{_prior_probabilities} );	
	return $random_state;		
}

# Subroutine to randomly sample a weighted distribution conditioned on the 
# transition probabilities of the neighboring nodes of the given node.
sub sample_from_priors	{
	my ( $self, $weights ) = @_;
	my @states = @{ $self->{_states} };
	my $total = sum( @{$weights} );
	
	my $rand = rand($total);
	my $chosen = 0;
	my $limit = $weights->[$chosen];
	while ( $rand >= $limit )	{
		$chosen++;
		$limit += $weights->[$chosen];
	}
	my $selected_state = $states[$chosen];
	return $states[$chosen]->{_name};
}

# Advances the model to the next state from a given state by randomly sampling from a weighted distribution
sub next_state		{
	my ( $self, $name ) = @_;	
	my $current_state = $self->_nameToState( $name );	
	my $index = $self->_stateToIndex( $current_state );
	my @trPr = @{$self->{_adj}->[$index]};		# The ROW in _adj 
	my $next_state = $current_state->weighted_random_sample( \@trPr );
	return $next_state->{_name};
}

# Generates the next k states of the system
# Returns an array REFERENCE containing the k future states of the model
sub generate_states		{
	my ( $self, $current_state, $k ) = @_;
	my @future_states;
	
	my $i = 0;
	while ( $i < $k )		{
		my $next_state = $self->next_state( $current_state );
		push @future_states, $next_state;
		$current_state = $next_state;
		$i++;
	}
	return \@future_states;	
}

# Generates an observation for a given state in accordance with the emission probabilities for that state
sub observation_from_state	{
	my ( $self, $name ) = @_;
	my $index = $self->_nameToIndex($name);
	my $state = $self->_nameToState($name);
	
	# Get the column in the emissions matrix _edj for the state we were given
	my $col;
	for ( my $i=0; $i < scalar @{$self->{_observation_states}}; $i++ )	{
		push @{$col}, $self->{_edj}->[$i]->[$index];
	}
	
	my $observation = $state->weighted_observation_sample( $col );
	return $observation;
}

# Generate a set of size $k samples from the HMM
# Returns two array references, @observations and @state_sequence
# @observations is the sequence of observations made while traversing the model
# @state_sequence is the sequence of latent/hidden states in the Markov chain that produced the observations
sub generate_samples	{
	my ( $self, $k, $start ) = @_;
	
	# If the user told us where to start, then great, we are happy.
	# If not, choose a random starting state from the model
	$start = ( $start )? $start : $self->random_state ;	
	
	# Initialize the state sequence array with the first starting state
	my @state_sequence = ( $start );
	
	# Also initialize the observation sequence
	my @observations = ( $self->observation_from_state($start) );
	
	# The loop to generate the next series of states and observations
	my $current_state = $start;
	foreach my $i ( 2 .. $k )	{
		my $next_state = $self->next_state($current_state);
		push @state_sequence, $next_state;
		push @observations, $self->observation_from_state($current_state);
		$current_state = $next_state;
	}	
	return ( \@observations, \@state_sequence );
}



# Include several functions to generate various common distributions for priors


##########################################################################
# DISCRETE-TIME MARKOV CHAIN PROCESSES
##########################################################################

# helper function that executes a DFS search and returns a boolean true or false if node $v is reachable from node $u
sub _isAccessible	{
	my ( $self, $u, $v ) = @_;
	my $preorder = $self->DFS( $u );
	my %Elements = map { $_ => 1 } @{$preorder};
	if ( exists $Elements{$v} )	{
		return 1;
	}
	else		{
		return 0;
	}
}

# Helper function that returns 1 if a Markov model cannot be reduced (if all states are reachable from any other state)
# Returns 0 if not.
sub _isIrreducible	{
	my ( $self ) = @_;
	my %AllStates = ();
	foreach my $state ( @{$self->{_states}} )	{
		my $name = $self->_stateToName($state);
		$AllStates{$name} = 1;
	}
	
	for ( my $u=0; $u < scalar @{ $self->{_states} }; $u++ )	{
		my $uname = $self->_stateToName( $self->{_states}->[$u] );
		my @reachable_nodes = @{ $self->DFS($uname) };
		
		# Check if @reachable nodes contains the right number of nodes, 
		# Then confirm that each node is in the hash
		if ( scalar @reachable_nodes == scalar @{ $self->{_states} } )	{
			for ( my $v=0; $v < scalar @reachable_nodes; $v++ )	{
				return 0 if ( not exists($AllStates{$reachable_nodes[$v]}) );
			}
		}
		else		{
			return 0;
		}
	}
	return 1;
}


# Depth-first search of the markov chain
# Returns an array reference containing either the pre- or post-ordered names of the nodes (tree-edges only, not cross/forward/back edges)
sub DFS		{
	my ( $self, $src, $type ) = @_;
	my @queue = ($src);
	my %Seen = ();
	my @preorder;
	my @postorder;
	$type = ( $type && $type =~ /post/ )? "post" : "pre";
	my @visited_children = ($src);
	
	while( scalar( @queue ) > 0 )	{
		my $vertex = pop( @queue ); 
		push @preorder, $vertex if ( not exists($Seen{$vertex}) );			
		
		# Get the neighbors of $vertex --> remember: neighbors is a Set!
		my $vstate = $self->_nameToState( $vertex );
		my $neighbors = $vstate->get_neighbors;
		
		# Add the neighbors to the queue and update some information about the current node
		foreach ( $neighbors->members() )	{
			print join(", ", @visited_children), "\n";
			push @visited_children, $_;
			if ( exists $Seen{$_} )	{
				# Can identify cycles, cross/back/forward edges here!
				pop @visited_children;
				print join(", ", @visited_children), "\n";
				next;
			}
			
			push @queue, $_;
		}
		
		# Add the current vertex to %Seen
		push @postorder, $vertex if ( scalar @queue == 1 );
		$Seen{$vertex} = 1;
		
			
	}
	
	# Return the ordering requested
	( $type eq "post" )? return \@postorder : return \@preorder; 			 # NOTE: 10-10-2019 MHS: post-order return currently not implemented, so only use pre-order
}


# Shamelessly jacked and refactored from somewhere on Google
# Subroutine for a BREADTH first search of a graph structure
sub BFS		{
	my ( $self, $src ) = @_;
	my @queue = ($src);
	my @bfs;
	my %Seen = ();
	
	while( scalar( @queue ) > 0 )	{
		my $vertex = shift( @queue ); 
		push @bfs, $vertex if ( not exists($Seen{$vertex}) );
			
		# Get the neighbors of $vertex --> remember: neighbors is a Set!
		my $vstate = $self->_nameToState( $vertex );
		my $neighbors = $vstate->get_neighbors;
		foreach ( $neighbors->members() )	{
			next if ( exists $Seen{$_} );
			push @queue, $_;
		}
			
		# Update the current vertex information
		$Seen{$vertex} = 1;
	}
	return \@bfs;
}

# Floyd-Warshall algorithm for computing all pairs shortest paths
sub APSP_floyd_warshall		{
	my ( $self ) = @_;
	
	# Get the corrected diagonals _adj for this algorithm
	my @M = @{$self->get_adj};
	my $n = scalar @M;
	
	# Initialize an empty HoH to store the length of the shortest paths
	# AND an empty matrix to hold traceback data
	my %FloydShortestPaths = ();
	my %Traceback = ();
	
	# Save off an enumeration of the nodes so that we can map them back later
	my @order = @{ $self->get_states };
	my %Indices = ();
	for ( my $i = 0; $i < scalar @order; $i++ )	{
		my $node = $self->_stateToName($order[$i]);
		$Indices{$i} = $node;
	}
	
	# Set all cells in the traceback matrix to 0
	for ( my $i=0; $i < $n; $i++ )	{
		for ( my $j=0; $j < $n; $j++ )	{
			$Traceback{$Indices{$i}}{$Indices{$j}} = 0;
		}
	}
	
	foreach my $i ( 0..$n-1 )	{
		foreach my $j ( 0..$n-1 )	{
			$Traceback{$Indices{$i}}{$Indices{$j}} = $i;
			
			if ( $i != $j && $M[$i][$j] == 0 )	{
				$Traceback{$Indices{$i}}{$Indices{$j}} = "inf";
				$M[$i][$j] = "inf";
			}
		}
	}
	
	# Set up the Floyd-Warshall nested loops:
	# Use the indices we set up in the enumeration to act as the keys for the ShortestPaths HoH
	foreach my $k ( 0..$n-1 )	{
		my $kX = $Indices{$k};
		foreach my $i ( 0..$n-1 )	{
			my $iX = $Indices{$i};
			foreach my $j ( 0..$n-1 )	{
				my $jX = $Indices{$j};
				if ( $M[$i][$j] > $M[$i][$k] + $M[$k][$j] )	{
					$FloydShortestPaths{$iX}{$jX} = $M[$i][$k] + $M[$k][$j];
					$Traceback{$iX}{$jX} = $Traceback{$kX}{$jX};
					$M[$i][$j] = $M[$i][$k] + $M[$k][$j];
				}
				else		{
					$FloydShortestPaths{$iX}{$jX} = $M[$i][$j];
					$Traceback{$iX}{$jX} = $Traceback{$iX}{$jX};
				}				
			}
		}
	}
	return ( \%FloydShortestPaths, \%Traceback );
}

############################## FLOYD WARSHALL UTILITY SUBROUTINES #####################################################
# Traces back the sequence of nodes from source U --> target V using the Traceback matrix built in the Floyd-Warshall subroutine
# Input args are ( @Traceback AoAref rom FW algorithm, vertex U, vertex V, and a path arrayref, which wil typically be empty when originally calling this routine )
sub get_path	{
	my ( $traceback, $i, $j, $path ) = @_;	
	my @order = sort keys %{$traceback};

	if ( $i eq $j )	{
		push @{$path}, $i;
	}
	elsif ( $traceback->{$i}->{$j} eq "inf" ) 	{
		push @{$path}, "-";
	}
	else		{
		my $k = $order[ $traceback->{$i}->{$j} ];
		get_path( $traceback, $i, $k, $path );
		push @{$path}, $j;
	}
	return $path;
}

# Returns the period of a given state in the Markov chain
sub get_period	{
	my ( $self, $state ) = @_;
	my $bellmanford = $self->bellman_ford($state);
	my @paths = values %{$bellmanford};
	return find_gcd( \@paths );
}

# Find the GCD of a list of numbers
sub find_gcd 	{
	my $list = @_;
	my $gcd = gcd_euclid( $list->[0], $list->[1] );		# The first two elements
	if ( scalar @{$list} > 2 )	{
		for ( my $i=2; $i < scalar @{$list}; $i++ )	{
			next if ( $list->[$i] eq "inf" || $list->[$i] eq "nan" );
			$gcd = gcd_euclid( $gcd, $list->[$i] );
		}
	}
	return $gcd;
}


# Helper function for computing periodicity
sub gcd_euclid 	{
  	my ($a, $b) = @_;
  	($a,$b) = ($b,$a) if $a > $b;
  	while ($a) {
    		($a, $b) = ($b % $a, $a);
  	}
  	return $b;
}

# Returns boolean true if the Markov chain is aperiodic, and false if it is periodic
sub _isAperiodic	{
	my ( $self ) = @_;
	
	my @periods;
	for ( my $u=0; $u < scalar @{ $self->{_states} }; $u++ ) 	{
		my $name = $self->_stateToName( $self->{_states}->[$u] );
		my $period = $self->get_period($name);
		return 0 if ( $period != 1 );
	}
	return 1;
}

# Checks if a given state is transient or not, returns boolean true if yes, false if no.
# If a state is transient, then by definition, once the system leaves this state, it will not be able to come back.
sub _isTransient	{
	my ( $self, $state ) = @_;
	my $index = $self->_nameToIndex($state);
	my $sum = 0;
	for ( my $c=0; $c < scalar @{$self->{_adj}}; $c++ )	{
		$sum += $self->{_adj}->[$c]->[$index];
	}
	( $sum < 1 )? return 1 : return 0;
	
}

# Checks if a given state in the Markov chain is absorbing, meaning 
# that once we arrive at this state, it is not possible to transition out of it
sub _isAbsorbing	{
	my ( $self, $state ) = @_;
	my $index = $self->_nameToIndex($state);
	my @row = @{ $self->{_adj}->[$index] };
	
	for ( my $r=0; $r < scalar @row; $r++ )	{
		if ( $row[$r] > 0  )	{
			if ( $r == $index )		{
				next;
			}
			else		{
				return 0; 
			}
		}
		else		{
			return 1;
		}
	}
}

# Utility subroutine
# Convert the _adj matrix built into our model into an HoH, since some subroutines here operate more easily on HoH (and were originally written as such)
# Returns a reference to a HoH
sub AoA2HoH	{
	my ( $self ) = @_;
	my %HoH = ();
	for ( my $i=0; $i < scalar @{$self->{_states}}; $i++ )	{
		my $name1 = $self->_stateToName( $self->{_states}->[$i] );
		for ( my $j=0; $j < scalar @{$self->{_states}}; $j++ )	{
			my $name2 = $self->_stateToName( $self->{_states}->[$j] );
			$HoH{$name1}{$name2} = $self->{_adj}->[$i]->[$j];
		}
	}
	return \%HoH;
}

# Print a matrix in a nice format --> does not assume a square matrix
sub print_matrix	{  
	my ( $self, $matrix, $sep ) = @_;
	$sep = ( $sep )? $sep : "\t";
	for ( my $u=0; $u < scalar @{$matrix}; $u++ )	{
		print join("$sep", @{$matrix->[$u]}), "\n";
	}
}

sub max	{
	my ( $self, $array ) = @_;
	my $max = $array->[0];
	for ( my $i=0; $i < scalar @{$array}; $i++ )	{
		$max = ( $array->[$i] > $max )? $array->[$i] : $max;
	}
	return $max;
}


##########################################################################
# DYNAMIC PROGRAMMING ALGORITHMS FOR STATE INFERENCES AND LEARNING
##########################################################################

# The forward algorithm for state inference
# Here we are given some observation data and want to predict the most probable state sequence traversing the HMM 
# which produced that set of observational sequence.
# Accepts a list of observation sequences of legnth T, and the state graph of length N
# Returns the probability for the observations to occur (as a float)

# function FORWARD( observations of len T, state-graph of len N), returns forward-prob
sub forward		{
	my ( $self, $observations ) = @_;
	my $priors = $self->{_prior_probabilities};
	my $transition = dclone $self->{_adj};					
	my $emission = dclone $self->{_edj};
	my $n_states = scalar @{$self->{_states}};
	my $prob = 0.0;
	
	# Initialize a 2D probability matrix fwd[N+2,T], where states 0 and F and non-emitting ( state 0 is the prior, and state F is the posterior )
	# for each state s from 1 to N do
	#     fwd[s,1] = adj[0,s] * edj[s, observations[1] ]		# Initialize, where adj[0] = priors 
	# done
	# (the following code is modified slightly to account for Perl's 0-indexing)
	my $fwd = [];	
	my $obs0 = $self->_obsNameToIndex( $observations->[0] );		# The edj index of the starting observation state
	for ( my $s=0; $s < $n_states; $s++ )	{
		$fwd->[$s]->[0] = $emission->[$obs0]->[$s];	 # * $priors->[$s];		# Note that we are not using priors here since these represent a starting state 0, which this model does not use
	}
	
	# Iteration steps:
	# for each time step t from 2 to T do
	# 	  for each state s from 1 to N do
	# 		  fwd[s,t] = sum( fwd[s',t-1] * adj[s',s] * edj[s, observations[t] ]  )		--> where s' = previous state from state s
	#	  done
	# done
	for ( my $t=1; $t < scalar @{$observations}; $t++ )		{
		# Annoying setups that we must do, since perl doesn't allow dynamic access to indicies and bc we want readable code
		my $t1 = $t - 1; 
		my $obst = $self->_obsNameToIndex( $observations->[$t] );
		
		# The inner loop, for each state s from 1 to N, do
		#      fwd[s,t] = sum( fwd[s',t-1] * adj[s',s] * edj[s, observations[t] ] ), where s' = the previous state from s
		for ( my $s=0; $s < $n_states; $s++ )	{
			for ( my $sprime=0; $sprime < $n_states; $sprime++ )	{
				$fwd->[$s]->[$t] += $fwd->[$sprime]->[$t1] * $transition->[$sprime]->[$s] * $emission->[$obst]->[$s];
			}
		}
	}
	
	# Termination step -- the final probability = sum of the last column in the forward trellis
	# fwd[qF,T] = total sum( fwd[s,T] * adj[s,qF] ), where qF is the final/end state of the model and non-emitting 
	# (here, we don't have a final end state, so we don't include the adj[s,qF] multiplier)
	my $r = scalar @{$observations} - 1;
	for ( my $s=0; $s < $n_states; $s++ )	{
		$prob += $fwd->[$s]->[$r];
	}
		
	# return fwd[qF,T]
	$prob = sprintf("%.5f", $prob);
	return ( $prob, $fwd );
}

# function BACKWARD( observations of len T, state-graph of len N), returns forward-prob
sub backward	{
	my ( $self, $observations ) = @_;
	my $transition = dclone $self->{_adj};					
	my $emission = dclone $self->{_edj};
	my $n_states = scalar @{$self->{_states}};
	my $prob = 0.0;
	my $T = scalar @{$observations};
	
	# Initialization step
	my $bkw = [ ];
	my $obs0 = $self->_obsNameToIndex( $observations->[-1] );	
	for ( my $s=0; $s < $n_states; $s++ )	{
		$bkw->[$s]->[$T-1] = $emission->[$obs0]->[$s];				# This isnt really useful in our model since this is the FINAL state, and our model as currently implemented, doesnt use one
	}
	
	# Recursion steps
	for ( my $t=2; $t <= $T; $t++ )		{
		my $t1 = $T - $t;
		my $obst1 = $self->_obsNameToIndex( $observations->[$t1] );
		for ( my $s=0; $s < $n_states; $s++ )	{
			for ( my $sprime=0; $sprime < $n_states; $sprime++ )	{
				$bkw->[$s]->[$t1] += $bkw->[$sprime]->[$t1+1] * $transition->[$sprime]->[$s] * $emission->[$obst1]->[$s];
			}
		}
	}
	
	
	# Termination step -- the final probability = sum of the first column in the backward trellis
	for (my $s=0; $s < $n_states; $s++ )	{
		$prob += $bkw->[$s]->[0];
	}
	
	# return bkw[qF,T]
	$prob = sprintf("%.5f", $prob);
	return ( $prob, $bkw );
}

# Forward-backward algorithm
sub forward_backward	{
	my ( $self, $observations ) = @_;
	my $priors = $self->{_prior_probabilities};
	my $transition = dclone $self->{_adj};					
	my $emission = dclone $self->{_edj};
	my $n_states = scalar @{$self->{_states}};
	my $T = scalar @{$observations};
	
	########################################
	# The forward part of the algorithm 
	my $fwd = [ ];
	for ( my $t=0; $t < scalar @{$observations}; $t++ )	{
		my $obst = $self->_obsNameToIndex( $observations->[$t] );
		
		for ( my $s=0; $s < $n_states; $s++ )	{
			# Base case for forward part
			if ( $t == 0 )		{
				$fwd->[$s]->[0] = $emission->[$obst]->[$s];
			}
			else		{
				for ( my $sprime=0; $sprime < $n_states; $sprime++ )	{
					my $t1 = $t - 1;
					$fwd->[$s]->[$t] += $fwd->[$sprime]->[$t1] * $transition->[$sprime]->[$s];
				}
			}
			$fwd->[$s]->[$t] = $emission->[$obst]->[$s] * $fwd->[$s]->[$t];
		}
	}
	my $p_fwd = 0;
	for ( my $s=0; $s < $n_states; $s++ )	{
		$p_fwd += $fwd->[$s]->[-1];
	}
	$p_fwd = sprintf("%.5f", $p_fwd);
	
	########################################
	# The backward part of the algorithm
	# Initialization step
	my $bkw = [ ];
	my $obs0 = $self->_obsNameToIndex( $observations->[-1] );	
	for ( my $s=0; $s < $n_states; $s++ )	{
		$bkw->[$s]->[$T-1] = $emission->[$obs0]->[$s];				# This isnt really useful in our model since this is the FINAL state, and our model as currently implemented, doesnt use one
	}
	for ( my $t=2; $t <= $T; $t++ )		{
		my $t1 = $T - $t;
		my $obst1 = $self->_obsNameToIndex( $observations->[$t1] );
		for ( my $s=0; $s < $n_states; $s++ )	{
			for ( my $sprime=0; $sprime < $n_states; $sprime++ )	{
				$bkw->[$s]->[$t1] += $bkw->[$sprime]->[$t1+1] * $transition->[$sprime]->[$s] * $emission->[$obst1]->[$s];
			}
		}
	}
	
	
	# Termination step -- the final probability = sum of the first column in the backward trellis
	my $p_bkw = 0;
	for (my $s=0; $s < $n_states; $s++ )	{
		$p_bkw += $bkw->[$s]->[0];
	}
	$p_bkw = sprintf("%.5f", $p_bkw);
	
	########################################
	# Merge the two parts
	my $posterior = [ ];
	for ( my $t=0; $t < scalar @{$observations}; $t++ )	{
		for ( my $s=0; $s < $n_states; $s++ )	{
			$posterior->[$s]->[$t] = $fwd->[$s]->[$t] * $bkw->[$s]->[$t] / $p_fwd;
		}
	}
	
	# Sanity check
	if ( $p_fwd == $p_bkw )	{
		return ( $p_fwd, $p_bkw, $posterior );
	}
	else		{
		warn " --- WARNING: HMM::forward_backward() error: the calculated forward and backward probabilities are not equal!  Treat results with extreme caution!\n";
		return ( $fwd, $bkw, $posterior );		# Returns the FWD and BKW matrices
	}
}

# function VITERBI --> returns the viterbi probability and the most probable sequence of hidden states which produced a given set of observations
# This is the DECODING algorithm, not the unsupervised learning algorithm!
sub viterbi		{
	my ( $self, $observations ) = @_;
	my $priors = $self->{_prior_probabilities};
	my $transition = dclone $self->{_adj};					
	my $emission = dclone $self->{_edj};
	my $n_states = scalar @{$self->{_states}};
	my $vprob = 0;	
	
	# Initialization steps:
	# Initialize a path probability matrix $viterbi[N+2,T], where the first column in the trellis table is the prior prob for state s
	# for each state s from 1 to N, do
	#     viterbi[s,0] = prior[s] * emission[s, obs0]
	#	  backpointer[s,0] = 0
	# done
	my $viterbi = [ ];
	my $backpointer = [ ];
	my $obs0 = $self->_obsNameToIndex( $observations->[0] );		# The edj index of the starting observation state
	for ( my $s=0; $s < $n_states; $s++ )	{
		$viterbi->[$s]->[0] = $emission->[$obs0]->[$s];	# * $priors->[$s]   See note in Forward algorithm
		$backpointer->[$s]->[0] = $s;
	}
	
	# Recursion steps:
	# for each time step t from 2 to T, do
	#     for each state s from 1 to N, do
	#		  for each state s' from 1 to N, do
	#	          viterbi[s,t] = max{ viterbi[s',t-1] * adj[s',s] * emission[observation at time t] }
	#             backpointer[s,t] = argmax{ viterbi[s',t-1] * adj[s',s]  }
	#		  done
	#	  done
	#  done
	for ( my $t=1; $t < scalar @{$observations}; $t++ )		{
		# Annoying setups that we must do, since perl doesn't allow dynamic access to indicies and bc we want readable code
		my $t1 = $t - 1; 
		my $obst = $self->_obsNameToIndex( $observations->[$t] );
		
		# The inner loops: 
		# for each state s from 1 to N, do
		# 		for each state s' from 1 to N, do
		#	         viterbi[s,t] = max{ viterbi[s',t-1] * adj[s',s] * emission[observation at time t, s] }
		#             backpointer[s,t] = argmax{ viterbi[s',t-1] * adj[s',s]  }
		for ( my $s=0; $s < $n_states; $s++ )	{
			my $max_tr_prob = 0;
			my $max_b_prob = 0;
			my $prev_state_index = $s;										
			
			for ( my $sprime=0; $sprime < $n_states; $sprime++ )	{			
				my $tr_prob = $viterbi->[$sprime]->[$t1] * $transition->[$sprime]->[$s] * $emission->[$obst]->[$s];
				my $b_prob = $viterbi->[$sprime]->[$t1] * $transition->[$sprime]->[$s];
				
				# Max for viterbi probability
				if ( $tr_prob > $max_tr_prob )	{
					$max_tr_prob = $tr_prob;
				}
				# Argmax for backpointer
				if ( $b_prob > $max_b_prob )	{
					$prev_state_index = $sprime;
				}
			}
			$viterbi->[$s]->[$t] = $max_tr_prob;
			$backpointer->[$s]->[$t] = $prev_state_index;
		}
	}
	
	# Termination steps:
	# viterbi[qF,T] = max{ viterbi[s,T] * adj[s,qF] }   
	# --> in English: the maximum value in the last column of the viterbi trellis (times) the transition probability to the end state, if one exists
	my $prob_column;
	for ( my $c=0; $c < scalar @{$viterbi}; $c++ )	{
		$prob_column->[$c] = $viterbi->[$c]->[-1];
	}
	$vprob = $self->max( $prob_column );
	
	# backpointer[qF,T] = argmax{ viterbi[s,T] * adj[s,qF] }
	my $backtrace = [ ];
	my $previous_state_index;
	
	# Get the first element in the backtrack
	for ( my $s=0; $s < scalar @{$prob_column}; $s++ )	{
		if ( $prob_column->[$s] == $vprob )	{
			unshift @{$backtrace}, $self->_indexToName( $s );
			$previous_state_index = $s;
			last;
		}
	}
     
     # Continue following the backtrack until we reach the first observationm which will be column 1 in the matrix (column 0 is the priors) 
     # We will go backwards through the viterbi trellis, getting the maximum value in each column to construct the backtrace
	for ( my $t = 2; $t <= scalar @{$observations}; $t++ )	{		
		my $u = scalar @{$observations} - $t;		# We already have the first column, so start with the second to last
		
		# Get the column and the max value in it
		my $column;
		for ( my $c=0; $c < scalar @{$viterbi}; $c++ )	{
			$column->[$c] = $viterbi->[$c]->[$u];
		}
		my $col_max = $self->max( $column );
		
		# Add this to the backtrace
		for ( my $s=0; $s < scalar @{$column}; $s++ )	{
			next if ( $transition->[$previous_state_index]->[$s] == 0 );
			if ( $column->[$s] == $col_max )	{
				unshift @{$backtrace}, $self->_indexToName( $s );
				$previous_state_index = $s;
				last;
			}
		}
	}
	
	# Return the maximum probability, and the backtrace path by following backpointers to hidden states back in time from backpointer[qF,T]
	$vprob = sprintf("%.5f", $vprob);
	return ( $vprob, $backtrace );
}

# the Baum-Welch EM algorithm for unsupervised learning 
# This updates the Transition and Emission matrices during each iteration.
sub baum_welch_learn		{
	my ( $self, $observations, $n_iter ) = @_;
	my $transition = dclone $self->{_adj};					
	my $emission = dclone $self->{_edj};
	my $T = scalar @{$observations};
	
	# Sanity check
	$n_iter = ( $n_iter && int($n_iter) >= 1 )? $n_iter : 100;
	
	# Convert the observation array into a list of state indicies, V
	my $V = [ ];
	foreach my $obs ( @{$observations} )	{
		push @{$V}, $self->_obsNameToIndex( $obs );
	}
	
	# The main EM loop:
	for ( my $n=0; $n < $n_iter; $n++ )		{
		#########################################################
		# EXPECTATION:
		# Get our forward and backward probabilities for this iteration --> we want the trellises from these algorithms
		my ( $fprob, $alpha ) = $self->forward( $observations );
		my ( $bprob,  $beta ) = $self->backward( $observations );
	
		# Initalize a 3-D matrix $xi (careful with this many dimensions!), where all cells are init to 0
		my $xi = [ ];
		for ( my $i=0; $i < scalar @{$transition}; $i++ )	{
			for ( my $j=0; $j < scalar @{$transition}; $j++ )	{
				for ( my $k=0; $k < ($T-1); $k++ )	{
					$xi->[$i]->[$j]->[$k] = 0;
				}
			}
		}
		
		# For t in range T-1 ==> will be T-2 due to Perl indexing
		for ( my $t=0; $t < ($T-2); $t++ )	{
			my $denominator = $self->dot( $self->multiply_vectors( $self->dot($alpha->[$t], $a), $self->column_slice($b, $V->[$t+1]) ), $beta->[$t+1]);		 # Oof, thats ugly
			for ( my $i=0; $i < scalar @{$transition}; $i++ )	{
				my $numerator = $self->multiply_vectors( $self->multiply_vectors($alpha->[$t]->[$i], $a->[$i]), $self->multiply_vectors($self->column_slice($b, $V->[$t+1]), $beta->[$t+1]) );
				for ( my $x=0; $x < scalar @{$xi->[0]}; $x++ )	{
					$xi->[$i]->[$x]->[$t] = $self->divide_vectors( $numerator, $denominator );		# $xi is now a 4D matrix
				}
			}
		}
		
		#########################################################
		# MAXIMIZATION:
		# Update the transition matrix:
		# Sum along axis 1 of $xi ( along the columns, which produces a row sum for each successive dimension )
		# Save this off, we will use it a few more times this iteration
		my $gamma = $self->sumAxis1( $xi );

		# Now, sum along axis 2 of $xi, then divide by the combined sum along axis1 of $gamma
		my $xiAxis2_sum = $self->sumAxis2( $xi );

		# The denominator for the above calculation:
		my $reshape_gamma = [ ];
		for ( my $i=0; $i < scalar @{$gamma}; $i++ )	{
			for ( my $j=0; $j < scalar @{$gamma->[$i]}; $j++ )	{
				$reshape_gamma->[$i] += $gamma->[$i]->[$j];
			}
		}
		
		# Finally, update the transition matrix
		my $transition = [ ];
		for ( my $i=0; $i < scalar @{$xiAxis2_sum}; $i++ )	{
			for ( my $j=0; $j < scalar @{$reshape_gamma}; $j++ )	{
				$transition->[$i]->[$j] = sprintf("%1.8f", ($xiAxis2_sum->[$i]->[$j] / $reshape_gamma->[$i]));
			}
		}
		
		# Add additional T'th element in gamma, which is the column sum of the last observation set ( = $xi[:,:,T-2] )
		for ( my $i=0; $i < scalar @{$xi}; $i++ )	{
			for ( my $j=0; $j < scalar @{$xi}; $j++ )	{
				$gamma->[$j]->[$T-1] += $xi->[$i]->[$j]->[$T-2];
			}
		}
		
		#########################################################
		# Update the emission matrix
		# Get the sum along the rows of the updated $gamma matrix (axis=1)
		my $denominator = [ ];
		$denominator = $self->sumAxis1( $gamma );
	
		for ( my $i=0; $i < scalar @{$emission}; $i++ )		{
			for ( my $l=0; $l < scalar @{$emission->[0]}; $l++ )	{		# The number of columns in the emission matrix
				# From the sequence of observations V, get the count of occurences where V[$i] == K, then multiply that by the correct column in gamma ($gamma[$K])
				# b[:, l] = np.sum(gamma[:, V == l], axis=1)
				for ( my $m=0; $m < scalar @{$V}; $m++ )	{
					$emission->[$i]->[$l] += $gamma->[$i]->[$m] if ( $V->[$m] == $l );
				}
			}
		}

		# Update the emissions with the calculated new probabilities
		for ( my $i=0; $i < scalar @{$emission}; $i++ )		{
			for ( my $j=0; $j < scalar @{$emission->[0]}; $j++ )	{
				$emission->[$i]->[$j] = sprintf("%1.8f", ($emission->[$i]->[$j] / $denominator->[$i]) );
			}
		}
	}
	
	# Update the transition and emission matrices in the main object
	$self->set_adj( $transition );
	$self->set_edj( $emission );
}

sub sumAxis0	{
	my ( $self, $matrix ) = @_;
	my $sumAxis0;
	for ( my $i=0; $i < scalar @{$matrix}; $i++ )	{
		for ( my $j=0; $j < scalar @{$matrix->[$i]}; $j++ )	{
			my $axis1 = $matrix->[$i]->[$j];
			for ( my $k=0; $k < scalar @{$axis1}; $k++ )	{
				$sumAxis0->[$j]->[$k] += $axis1->[$k];
			}
		}
	}
	return $sumAxis0;
}

# Will handle up to a 4D array
sub sumAxis1		{
	my ( $self, $matrix ) = @_;
	my $sumAxis1;
	for ( my $i=0; $i < scalar @{$matrix}; $i++ )	{
		for ( my $j=0; $j < scalar @{$matrix->[$i]}; $j++ )	{
			my $axis1 = $matrix->[$i]->[$j];
			# If we have a 3D matrix
			if ( ref($axis1) eq 'ARRAY' )	{
				for ( my $k=0; $k < scalar @{$axis1}; $k++ )	{
					if ( ref($axis1->[$k]) eq 'ARRAY' )	{
						$sumAxis1->[$i]->[$k] += $axis1->[$k]->[0];
					}
					else	{
						$sumAxis1->[$i]->[$k] += $axis1->[$k];
					}
				}
			}
			# Else we just have a 2D one and want the rowsums
			else	{
				$sumAxis1->[$i] += $matrix->[$i]->[$j];
			}
		}
	}
	return $sumAxis1;
}

# Calculates the sum of each column in a numerical matrix, return an array (requires at least a 3D matrix)
# Can handle up to a 4D matrix
sub sumAxis2		{
	my ( $self, $matrix ) = @_;
	my $sumAxis2;
	for ( my $i=0; $i < scalar @{$matrix}; $i++ )	{
		for ( my $j=0; $j < scalar @{$matrix->[$i]}; $j++ )	{
			my $axis1 = $matrix->[$i]->[$j];
			if ( ref($matrix->[$i]->[$j]) ne 'ARRAY' )	{
				warn "HMM::sumAxis2 ERROR --> I cant sum Axis 2 of a matrix that doesn't have three dimensions!\n";
				return 0;
			}
			for ( my $k=0; $k < scalar @{$axis1}; $k++ )	{
				if ( ref($axis1->[$k]) eq 'ARRAY' )	{		# 4 dimensions
					$sumAxis2->[$i]->[$j] += $axis1->[$k]->[0];
				}
				else	{
					$sumAxis2->[$i]->[$j] += $axis1->[$k];
				}
			}
		}
	}
	return $sumAxis2;
}

# Transpose a 1D "matrix" (an array) into a column-list (which is technically a 2D array in Perl) or vice versa
sub transpose1D		{
	my ( $self, $matrix ) = @_;
	my $transposed = [ ];
	
	# If we have a 1D column, we want to return a 1D list
	if ( scalar @{$matrix} == 1 && scalar @{$matrix->[0]} > 1 )	{
		for ( my $i=0; $i < scalar @{$matrix->[0]}; $i++ )	{
			$transposed->[$i] = $matrix->[0]->[$i];
		}
	}
	# We have a standard array and want to turn it into a column list
	else	{
		for ( my $i=0; $i < scalar @{$matrix}; $i++ )	{
			$transposed->[0]->[$i] = $matrix->[$i];
		}
	}
	return $transposed;
}

# Tranpose a 2D matrix
sub transpose2D		{
	my ( $self, $matrix ) = @_;
	my $transposed = [ ];
	for ( my $i=0; $i < scalar @{$matrix}; $i++ )	{
		for ( my $j=0; $j < scalar @{$matrix->[0]}; $j++ )	{
			$transposed->[$j]->[$i] = $matrix->[$i]->[$j];
		}
	}
	return $transposed;
}

# Tranpose a 3D matrix
sub transpose3D		{
	my ( $self, $matrix ) = @_;
	my $transposed = [ ];
	for ( my $i=0; $i < scalar @{$matrix}; $i++ )	{
		for ( my $j=0; $j < scalar @{$matrix->[0]}; $j++ )	{
			for ( my $k=0; $k < scalar @{$matrix->[0]->[0]}; $k++ )	{
				$transposed->[$k]->[$j]->[$i] = $matrix->[$i]->[$j]->[$k];
			}
		}
	}
	return $transposed;
}

# Additional matrix mathematical operations functions
# dot product for two 2D matrices
sub dot		{
	my ( $self,$matrix1, $matrix2 ) = @_;	
	my $dot = [ ];

	# If we are given a vector and a scalar multiplier, eg: 3 * [1,2,3], will return a vector (this is just basic multiplication of matrices)
	if ( looks_like_number($matrix1) || looks_like_number($matrix2) )	{
		if ( looks_like_number($matrix2) && not looks_like_number($matrix1) )	{
			for ( my $m=0; $m < scalar @{$matrix1}; $m++ )	{
				$dot->[$m] = ($matrix1->[$m] * $matrix2);
			}
		}
		elsif ( looks_like_number($matrix1) && not looks_like_number($matrix2) )	{
			for ( my $m=0; $m < scalar @{$matrix2}; $m++ )	{
				$dot->[$m] = ($matrix1 * $matrix2->[$m]);
			} 	
		}
		else	{
			warn "HMM:dot_product ERROR: you gave me two regular ass numbers and wanted me to give you a matrix back.  That ain't how math works!\n";
			return 0;
		}
	}
	# If we already have a 1xN and Nx1 case, we will only return a matrix with a single cell (we will just return the scalar in this case... might consider changing this in the future)
	elsif  ( ref($matrix1->[0]) ne 'ARRAY' && ref($matrix2->[0]) ne 'ARRAY' )	{
		if ( scalar @{$matrix1} != scalar @{$matrix2} )	{
			warn "HMM::dot product ERROR: The vectors don't have the same size!\n";
			return 0;
		}
		for ( my $m=0; $m < scalar @{$matrix1}; $m++ )	{
			$dot->[0] += ($matrix1->[$m] * $matrix2->[$m]);
		}
	}
	# If matrix1 only has 1 row, matrix 2 has multiple:
	elsif ( ref($matrix1->[0]) ne 'ARRAY' && ref($matrix2->[0]) eq 'ARRAY' )	{
		#print "DOT 2\n";
		my $m2_copy = $self->transpose2D( dclone( $matrix2 ) );
		if ( scalar @{$matrix1} != scalar @{$m2_copy->[0]} )	{
			warn "HMM::dot product ERROR: The vectors don't have the same size!\n";
			return 0;
		}
		for ( my $m=0; $m < scalar @{$matrix1}; $m++ )	{
			for ( my $n=0; $n < scalar @{$m2_copy->[0]}; $n++ )	{
				$dot->[$n] += ($matrix1->[$m] * $m2_copy->[$m]->[$n]);
			}
		}
	}
	# If matrix2 only has 1 row, matrix1 has multiple
	elsif ( ref($matrix2->[0]) ne 'ARRAY' && ref($matrix1->[0]) eq 'ARRAY' )	{
		( $matrix1, $matrix2 ) = ( $matrix2, $matrix1 );		# Swap them in this case
		my $m2_copy = $self->transpose2D( dclone( $matrix2 ) );
		if ( scalar @{$matrix1} != scalar @{$m2_copy->[0]} )	{
			warn "HMM::dot product ERROR: The vectors don't have the same size!\n";
			return 0;
		}
		for ( my $m=0; $m < scalar @{$matrix1}; $m++ )	{
			for ( my $n=0; $n < scalar @{$m2_copy->[0]}; $n++ )	{
				$dot->[$n] += ($matrix1->[$m] * $m2_copy->[$m]->[$n]);
			}
		}
	}
	# Otherwise we can assume that we have 2 2D matrices (MxN * NxP)
	else	{
		my $m2_copy = $self->transpose2D( dclone( $matrix2 ) );
		if ( scalar @{$matrix1->[0]} != scalar @{$m2_copy->[0]} )	{
			warn "HMM::dot product ERROR: The vectors don't have the same size!\n";
			return 0;
		}
		for ( my $m=0; $m < scalar @{$matrix1}; $m++ )	{
			for ( my $n=0; $n < scalar @{$matrix1->[0]}; $n++ )		{
				for ( my $p=0; $p < scalar @{$matrix2->[0]}; $p++ )	{
					$dot->[$m]->[$p] += ( $matrix1->[$m]->[$n] * $matrix2->[$n]->[$p] );
				}
			}
		}
	}
	return $dot;
}

# Get a column slice from a matrix --> careful with Perl indexing here!
# This is only intended to work on a 2D matrix, and return a 1D array --> note that the column slice is by default returned as transposed (ie. as a row, rather than a column)
sub column_slice	{
	my ( $self, $matrix, $start_index, $stop_index ) = @_;
	my $slice = [ ];
	
	# Sanity checks
	if ( not $matrix || not $start_index )	{
		warn "HMM::column_slice --> cant get a column slice without a starting index!\n";
		return 0;
	}
	$stop_index = ( $stop_index && $stop_index < scalar @{$matrix->[0]} && $stop_index > $start_index )? $stop_index : $start_index+1;
	
	# Get the requested slice
	for ( my $r=0; $r < scalar @{$matrix}; $r++ ) 	{
		for ( my $i=$start_index; $i < $stop_index; $i++ )	{
			$slice->[$r] = $matrix->[$r]->[$i];
		}
	}
	return $slice;
}

# Multiply 2 vectors together --> this is NOT a dot product!
sub multiply_vectors	{
	my ( $self, $vector_a, $vector_b ) = @_;
	my $multiplied = [ ];
	
	# Sanity checks
	$vector_a = ( ref($vector_a) eq 'ARRAY' && scalar @{$vector_a} == 1 )? $vector_a->[0] : $vector_a;
	$vector_b = ( ref($vector_b) eq 'ARRAY' && scalar @{$vector_b} == 1 )? $vector_b->[0] : $vector_b;
	
	# If we are given a vector and a scalar multiplier, eg: 3 * [1,2,3], will return a vector (this is just basic multiplication of matrices)
	if ( looks_like_number($vector_a) || looks_like_number($vector_b) )	{
		if ( looks_like_number($vector_b) && not looks_like_number($vector_a) )	{
			for ( my $m=0; $m < scalar @{$vector_a}; $m++ )	{
				$multiplied->[$m] = ($vector_a->[$m] * $vector_b);
			}
		}
		elsif ( looks_like_number($vector_a) && not looks_like_number($vector_b) )	{
			for ( my $m=0; $m < scalar @{$vector_b}; $m++ )	{
				$multiplied->[$m] = ($vector_a * $vector_b->[$m]);
			} 	
		}
		else	{
			warn "HMM:multiply_vectors ERROR: you gave me two regular ass numbers and wanted me to give you a vector back.  That ain't how math works!\n";
			return 0;
		}
	}
	else	{
		if ( scalar @{$vector_a} != scalar @{$vector_b} )	{
			warn "HMM::multiply_vectors ERROR: The vectors don't have the same size!\n";
			return 0;
		}
		for ( my $m=0; $m < scalar @{$vector_a}; $m++ )	{
			$multiplied->[$m] += ($vector_a->[$m] * $vector_b->[$m]);
		}
	}
	return $multiplied;
}

# Divide one vector by another --> uses 'nan' to accounting for division by zero.
sub divide_vectors	{
	my ( $vector_a, $vector_b ) = @_;
	my $divided = [ ];

	# Sanity checks
	$vector_a = ( ref($vector_a) eq 'ARRAY' && scalar @{$vector_a} == 1 )? $vector_a->[0] : $vector_a;
	$vector_b = ( ref($vector_b) eq 'ARRAY' && scalar @{$vector_b} == 1 )? $vector_b->[0] : $vector_b;
	
	# If we are given a vector and a scalar multiplier, eg: 3 * [1,2,3], will return a vector (this is just basic multiplication of matrices)
	if ( looks_like_number($vector_a) || looks_like_number($vector_b) )	{
		if ( looks_like_number($vector_b) && not looks_like_number($vector_a) )	{
			for ( my $m=0; $m < scalar @{$vector_a}; $m++ )	{
				if 		( $vector_a->[$m] == 0 )	{ $divided->[$m] = "nan";	}
				else	{	$divided->[$m] = ($vector_a->[$m] / $vector_b );	}
			}
		}
		elsif ( looks_like_number($vector_a) && not looks_like_number($vector_b) )	{
			for ( my $m=0; $m < scalar @{$vector_b}; $m++ )	{
				if 		( $vector_b->[$m] == 0 )	{ $divided->[$m] = "nan";	}
				else	{	$divided->[$m] = ($vector_b->[$m] / $vector_a );	}
			} 	
		}
		else	{
			warn "HMM:divide_vectors ERROR: you gave me two regular ass numbers and wanted me to give you a vector back.  That ain't how math works!\n";
			return 0;
		}
	}
	else	{
		if ( scalar @{$vector_a} != scalar @{$vector_b} )	{
			warn "HMM::divide_vectors ERROR: The vectors don't have the same size!\n";
			return 0;
		}
		for ( my $m=0; $m < scalar @{$vector_a}; $m++ )	{
			if 			( $vector_b->[$m] == 0 )	{ $divided->[$m] = "nan";	}
			else	{	$divided->[$m] += ($vector_a->[$m] / $vector_b->[$m]);	}
		}
	}
	return $divided;
}



# Back to where we started ;)
1;
