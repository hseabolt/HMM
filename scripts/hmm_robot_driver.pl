#!/usr/bin/perl

# markov_driver.pl
# Driver and testing code for HMM.pm using an example of a robot moving around a game board


use strict;
use warnings;
use HMM;

# Begin driver code
print "\n\n************************\n\n";

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
     [   0,    0,    0,    0,    0,    0,    0, 0.33,    0,    0,    0, 0.33, 0.33]
];

my $observation_states = [ "Top", "Bottom", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Top-Bottom", "Left-Right" ];		# Removed left only and right only, since no states have these 
my $emission_matrix = [
	[   0,   0, 1.0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],		# top only
	[   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 1.0,   0,   0 ],		# bottom only
	[ 1.0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],		# top and left
	[   0,   0,   0,   0, 1.0,   0,   0,   0,   0,   0,   0,   0,   0 ],		# top and right
	[   0,   0,   0,   0,   0,   0,   0,   0, 1.0,   0,   0,   0,   0 ],		# bottom and left
	[   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 1.0 ],		# bottom and right
	[   0,0.25,   0,0.25,   0,   0,   0,   0,   0,0.25,   0,0.25,   0 ],		# top and bottom
	[   0,   0,   0,   0,   0,0.33,0.33,0.33,   0,   0,   0,   0,   0 ]		# left and right
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
my @Adj = $model2->get_adj;
print "\n\nUpdated Adj: \n";
print join(", ", @{$model2->get_states}), "\n";
foreach my $u ( @Adj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Print the emissions matrix
my @Edj = $model2->get_edj;
print "\n\nEdj: \n";
print join(", ", @{$model2->get_observation_states}), "\n";
foreach my $u ( @Edj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}

my ( $sampled_observations, $hidden_states ) = $model2->generate_samples( 4 );
my $n = scalar @{$sampled_observations};


# Run the forward algorithm
print "\n\nForward Algorithm: \n";

print "Randomly sampled observations (n=$n): ", join(", ", @{$sampled_observations}), "\n";

my ( $forward_prob, $trellis ) = $model2->forward( $sampled_observations, "BM" );
print "Probability of state sequence ", join(", ", @{$sampled_observations}), ": $forward_prob%\n";
$model2->print_matrix( $trellis);

# Run the backward algorithm
print "\n\nBackward Algorithm: \n";

my $backward_prob = $model2->backward( $sampled_observations );
print "Probability of state sequence ", join(", ", @{$sampled_observations}), ": $backward_prob%\n";


# Run the Forward-Backward algorithm
print "\n\nForward-Backward Algorithm: \n";
my ( $fprob, $bprob, $posterior ) = $model2->forward_backward( $sampled_observations );
print "Forward-Backward probability of state sequence ", join(", ", @{$sampled_observations}), ": $fprob% Fwd -- $bprob% Bkw\n";
print "Posterior Trellis: \n";
for ( my $u=0; $u < scalar @{$posterior}; $u++ )	{
	print join("\t", @{$posterior->[$u]}), "\n";
}

# Run the Viterbi algorithm
print "\n\nViterbi Algorithm: \n";

my ( $viterbi_prob, $path ) = $model2->viterbi( $sampled_observations );
print "Viterbi probability of state sequence ", join(", ", @{$sampled_observations}), ": $viterbi_prob%\n";
print "Most probable path for this sequence of observations: ", join(" --> ", @{$path}), "\n";

# Run the Baum-Welch Learning algorithm --> generate a reasonably lengthy sequence of states, then retrain the model with Baum-Welch
print "\n\nBaum-Welch Learning Algorithm: \n";

my ( $sampled_observations, $hidden_states ) = $model2->generate_samples( 10000 );
$model2->baum_welch_learn( $sampled_observations, 100 );

# Print the adjacency matrix again
my @Adj = $model2->get_adj;
print "\n\nUpdated Adj: \n";
print join(", ", @{$model2->get_states}), "\n";
foreach my $u ( @Adj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}
print "\n\n";

# Print the emissions matrix
my @Edj = $model2->get_edj;
print "\n\nEdj: \n";
print join(", ", @{$model2->get_observation_states}), "\n";
foreach my $u ( @Edj )	{
	foreach my $v ( @{$u} )	{
		print join(", ", @{$v}), "\n";
	}
}



print "\n\n************************\n\n";
exit;
