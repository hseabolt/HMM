#!/usr/bin/perl

# package MarkovNode.pm

# Author: MH Seabolt
# Last Updated: 11-14-2019	

package MarkovNode;

require Exporter;
@ISA = qw(Exporter);
@EXPORT = qw(max min sum lg log10 pow round);			 #Import from other packages

# @INC libraries for my PC and Biolinux or HS custom classes 
use lib '/media/hunter/Data/scripts';
use lib 'D:\scripts';
use lib '/scicomp/home/ngr8/Biolinux/scripts';

use strict;
use warnings;
use List::Util qw( sum first shuffle );
use Carp;
our $AUTOLOAD;

################################################################################## 

# Class data and methods
# Attributes 
{
	my %_attribute_properties = (
		_name 			=> "",			# The name of the node (as a state name)
		_children			=> [ ],			# An anonymous array of the list of child nodes, where the element is the transition probability
		_observations		=> [ ],			# An anonymous array of the list of possible observations, where the element is emission probability
		_isObsState 		=> 0,			# An integer value representing whether this node is an observed or not in an  HMM state machine (doubles as a counter for the number of paths that terminate at this node)
		_metadata 		=> "",			# An empty (by default) string, which can contain any metadata that we wish to associate with this node
		_parent 			=> "",			# An optional pointer to set as the parent node for traversals of the graph structure 
										# 	--> NOTE: this is only intended for to set ONE node as the parent (which, of course, a node in a directed graph can have multiple parents).
										#             This was originally intended to set as the predecessor (ie. parent) for tracebacks through the model.
	);
	
	# Global variable counter
	my $_count = 0;
	
	# Return a list of all attributes
	sub _all_attributes	{
		keys %_attribute_properties;
	}
	
	# Return the default value for a given attribute
    	sub _attribute_default 	{
     	my($self, $attribute) = @_;
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

##########################################################################
# CONSTRUCTORS 
##########################################################################

# The contructor method
# Construct a new graph (my $node = TrieNode->new() );
# Returns a scalar reference to a
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
    	$class->_incr_count();
	return $self;
}

# AUTOLOAD getters and setters
sub AUTOLOAD	{
	my ( $self, $newvalue ) = @_;
	my ( $operation, $attribute ) = ( $AUTOLOAD =~ /(get|set)(_\w+)$/ );
	
	# Is method name legal?
	unless ( $operation && $attribute )	{
		croak "Method name $AUTOLOAD is not in the recognized form (get|set)_attribute\n";
	}
	unless ( exists $self->{$attribute} )	{
		croak "No such attribute $attribute exists in the class ", ref($self);	
	}
	
	# AUTOLOAD Getters
	no strict 'refs';		# Turn strict refs off
	if ( $operation eq 'get' )	{
		
		# Install the getter method in the symbol table
		*{$AUTOLOAD} = sub {
			my ( $self ) = @_;

		# Check what type of reference the attribute is (might be an ARRAY ref or a SCALAR)
		if ( ref($self->{$attribute}) eq 'ARRAY' )	{
				return @{$self->{$attribute}};
			}
			else	{
				return $self->{$attribute};
			}
		};
		
		# Turn back on strict refs :)
		use strict 'refs';
		
		# Return the attribute value
		if ( ref($self->{$attribute}) eq 'ARRAY' )	{
				return @{$self->{$attribute}};
		}
		else	{
			return $self->{$attribute};
		}
	}

	
	# AUTOLOAD Setters
	elsif ( $operation eq 'set' )	{
		
		# Set the new attribute value
		$self->{$attribute} = $newvalue;
		
		# Install the mutator definition in the symbol table
		*{$AUTOLOAD} = sub {
			my ( $self ) = @_;
			$self->{$attribute} = $newvalue;
		};
	}
	
	use strict 'refs'; 		# Turn strict refs back on
	
	# Return the attribute value
	return $self->{$attribute};	
}

# Subroutine to randomly sample a weighted distribution conditioned on the 
# transition probabilities of the neighboring nodes of the given node.
# $weights here is expected to be a reference to a row in the Markov{_adj} matrix.
sub weighted_random_sample	{
	my ( $self, $weights ) = @_;
	my @children = @{ $self->{_children} };
	my $total = sum( @{$weights} );
	
	my $rand = rand($total);
	my $chosen = 0;
	my $limit = $weights->[$chosen];
	while ( $rand >= $limit )	{
		#$chosen = 0 if ( $chosen > scalar @children );
		$chosen++;
		$limit += $weights->[$chosen];
	}
	return $children[$chosen];
}


# Subroutine to randomly sample a weighted distribution conditioned on the 
# emission probabilities of the observation states of the given node.
# $weights here is expected to be a reference to a row in the Markov{_edj} matrix.
sub weighted_observation_sample	{
	my ( $self, $weights ) = @_;
	my @children = @{ $self->{_observations} };
	my $total = sum( @{$weights} );
	
	my $rand = rand($total);
	my $chosen = 0;
	my $limit = $weights->[$chosen];
	while ( $rand >= $limit )	{
		$chosen++;
		$limit += $weights->[$chosen];
	}
	return $children[$chosen]->{_name};
}


# Get all the neighboring/adjcent states of a given node/vertex
# Returns a Set ( from Set::Scalar ) of the neighboring vertices.
sub get_neighbors	{
	my ( $self ) = @_;
	my $neighbors = Set::Scalar->new();
	foreach my $v ( @{$self->{_children}} )	{
		my $name = $v->{_name};
		$neighbors->insert( $name ) if ( $name );
	}
	return $neighbors;
}



# When an object is no longer being used, garbage collect it and adjust count of existing objects
sub DESTROY	{
	my ( $self ) = @_;
	$self->_decr_count();
}


# Last line in a package must always be 1 !
1;
