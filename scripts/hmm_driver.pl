#!/usr/bin/perl

# markov_driver.pl
# Driver and testing code for HMM.pm

# This example code models the weather (the latent variable which forms a Markov chain)
# by the observational states of muddy paws inside a house with no windows.

use strict;
use warnings;

use lib '/media/hunter/Data/scripts';
use lib '/scicomp/home/ngr8/Biolinux/scripts';
use HMM;

# Begin driver code
print "\n\n************************\n\n";

# Generate a set of latent/hidden states for the model
my @weather_states = ("sunny", "snowy", "rainy");
my $probs = [
	[ 0.7, 0.1, 0.2 ],
	[ 0.05, 0.7, 0.25 ],
	[ 0.25, 0.2, 0.55 ],
];

# Generate a set of observational states for the model
my @observed_states = ("muddy", "clean");
my $emitted = [
	[ 0.05, 0.60, 0.85 ],
	[ 0.95, 0.40, 0.15 ],
];

# Instantiate a new Markov object
my $model = HMM->new();


# Add the states to the model and set their names and emission probabilities
for ( my $u=0; $u < scalar @weather_states; $u++ )	{
	print "$weather_states[$u] -- Probs[u]: ", join(", ", @{$probs->[$u]}), "\n";
	$model->add_state( $weather_states[$u], $probs->[$u] );
}

# Add the observation states to the model
for ( my $u=0; $u < scalar @observed_states; $u++ )	{
	print "$observed_states[$u] -- Emits[u]: ", join(", ", @{$emitted->[$u]}), "\n";
	$model->add_observation_state( $observed_states[$u], $emitted->[$u] );
}

# Print the adjacency matrix
my @Adj = $model->get_adj;
print "\n\nAdj: \n";
print join(", ", @{$model->get_states}), "\n";
foreach my $u ( @Adj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Print the emissions matrix
my @Edj = $model->get_edj;
print "\n\nEdj: \n";
print join(", ", @{$model->get_observation_states}), "\n";
foreach my $u ( @Edj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";


# Add a 4th state
$model->add_state( "squanch", [ 0.2, 0.5, 0.1, 0.2 ], [ 0.5, 0.5 ] );

# Print the adjacency matrix
@Adj = $model->get_adj;
print "\n\nUpdated Adj: \n";
print join(", ", @{$model->get_states}), "\n";
foreach my $u ( @Adj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Print the emissions matrix
@Edj = $model->get_edj;
print "\n\nEdj: \n";
print join(", ", @{$model->get_observation_states}), "\n";
foreach my $u ( @Edj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Set the priors for the model --> just using equal priors here for example
$model->set_priors( [ 0.25, 0.25, 0.25, 0.25 ] );

# Get a random starting state
my $first_state = $model->random_state();
print "Randomly chosen starting first state: $first_state\n";

# Generate the next state of the model
my $second_state = $model->next_state( $first_state );
print "Second state: $second_state\n";

# Generate the next 10 states of the model
my @future_states = @{ $model->generate_states( $second_state, 10 ) };
print "Future state: $_\n" foreach ( @future_states );

# Delete a state
print "\n\n";
#$model->delete_state( "rainy" );
#print "Deleted rainy\n\n";

# Add a new observation state, swampy
$model->add_observation_state( "swamptastical", [ 0.0, 0.0, 0.0, 0.35 ] );

# Print the adjacency matrix again
@Adj = $model->get_adj;
print "\n\nUpdated Adj: \n";
print join(", ", @{$model->get_states}), "\n";
foreach my $u ( @Adj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Print the emissions matrix
@Edj = $model->get_edj;
print "\n\nEdj: \n";
print join(", ", @{$model->get_observation_states}), "\n";
foreach my $u ( @Edj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}

# Check if a node is accessible --> this should be a yes
my $check = $model->_isAccessible( "snowy", "sunny" );
if ( $check == 1 )	{	print "Yes, sunny is accessible from snowy\n";	}
else				{	print "Nope, sunny is not accessible from snowy\n";	}

# Check something that should be false
$check = $model->_isAccessible( "snowy", "rainy" );
if ( $check == 1 )	{	print "Yes, squanch is accessible from snowy\n";	}
else				{	print "Nope, squanch is not accessible from snowy\n";	}
print "\n\n";

# Generate an observation sequence
my ( $observations, $state_sequence ) = $model->generate_samples( 100 );
for ( my $i=0; $i < scalar @{$observations}; $i++ )	{
	print "Observation: $observations->[$i]   ---   Current Model State: $state_sequence->[$i] \n";
}

print "\n\n************************\n\n";

##########################################################################################################
# Reset the model for the robot problem
my $model2 = HMM->new();

my $states = [ "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen" ];
my $priors = [ 0.077, 0.077,   0.077,  0.077,  0.077, 0.077,   0.077,   0.077,  0.077, 0.077,    0.077,    0.077,     0.077  ];
my $transition_matrix = [
	[0.33, 0.33,    0,    0,    0, 0.33,    0,    0,    0,    0,    0,    0,    0],
     [0.33, 0.33, 0.33,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
     [   0, 0.25, 0.25, 0.25,    0,    0, 0.25,    0,    0,    0,    0,    0,    0],
     [   0,    0, 0.33, 0.33, 0.33,    0,    0,    0,    0,    0,    0,    0,    0],
     [   0,    0,    0, 0.33, 0.33,    0,    0, 0.33,    0,    0,    0,    0,    0],
     [0.33,    0,    0,    0,    0, 0.33,    0,    0, 0.33,    0,    0,    0,    0],
     [   0,    0, 0.33,    0,    0,    0, 0.33,    0,    0,    0, 0.33,    0,    0],
     [   0,    0,    0,    0, 0.33,    0,    0, 0.33,    0,    0,    0,    0, 0.33],
     [   0,    0,    0,    0,    0, 0.33,    0,    0, 0.33, 0.33,    0,    0,    0],
     [   0,    0,    0,    0,    0,    0,    0,    0, 0.33, 0.33, 0.33,    0,    0],
     [   0,    0,    0,    0,    0,    0,    0,    0,    0, 0.33, 0.33, 0.33,    0],
     [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0.33, 0.33, 0.33],
     [   0,    0,    0,    0,    0,    0,    0, 0.33,    0,    0,    0, 0.33, 0.33],
];

my $observation_states = [ "Top", "Left", "Right", "Bottom" ];
my $emission_matrix = [
	[ 1.0, 1.0, 1.0, 1.0, 1.0,   0,   0,   0,   0, 1.0,   0, 1.0,   0 ],		# Have a wall on the top
	[ 1.0,   0,   0,   0,   0, 1.0, 1.0, 1.0, 1.0,   0,   0,   0,   0 ],		# Have a wall on the left
	[   0,   0,   0,   0, 1.0, 1.0, 1.0, 1.0,   0,   0,   0,   0, 1.0 ],		# Have a wall on the right
	[   0, 1.0,   0, 1.0,   0,   0,   0,   0, 1.0, 1.0, 1.0, 1.0, 1.0 ],		# Have a wall on the bottom
];

# Add the states to the new model
for ( my $i=0; $i < scalar @{$states}; $i++ )	{
	$model2->add_state( $states->[$i], $transition_matrix->[$i], $priors->[$i] );
}

# Add the observation states and the emission probabilities
for ( my $j=0; $j < scalar @{$observation_states}; $j++ )	{
	$model2->add_observation_state( $observation_states->[$j], $emission_matrix->[$j] );
}


# Print the adjacency matrix again
@Adj = $model2->get_adj;
print "\n\nUpdated Adj: \n";
print join(", ", @{$model2->get_states}), "\n";
foreach my $u ( @Adj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Print the emissions matrix
@Edj = $model2->get_edj;
print "\n\nEdj: \n";
print join(", ", @{$model2->get_observation_states}), "\n";
foreach my $u ( @Edj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}

exit;

# Run the forward algorithm
print "\n\nForward Algorithm: \n";

my $sequence_of_observations = [ "One", "One", "One", "One" ];		# Staying in the same place
my $forward_prob = $model2->forward( $sequence_of_observations );
print "Probability of staying in state 'One' for 4 moves: $forward_prob\n";

$sequence_of_observations = [ "One", "Ten", "Eight", "Six" ];		# Should be zero since we cant jump from state 1 to state 10
$forward_prob = $model2->forward( $sequence_of_observations );
print "Probability of state sequence 'One -> Ten -> Eight -> Six': $forward_prob\n";

# Run the Viterbi algorithm
print "\n\nViterbi Algorithm: \n";
$sequence_of_observations = [ "swamptastical", "swamptastical", "swamptastical", "swamptastical" ];		# Staying in the same place
my ( $viterbi_prob, $path ) = $model2->viterbi( $sequence_of_observations );
print "Viterbi probability of staying in state 'swamptastical' for 4 moves: $viterbi_prob\n";
print "Most probable path for this sequence of observations: ", join(" --> ", @{$path}), "\n";


print "\n\n************************\n\n";
exit;
