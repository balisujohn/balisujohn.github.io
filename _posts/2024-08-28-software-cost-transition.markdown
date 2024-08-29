---
layout: post
title: The Software Cost Transition
date: 2024-8-28 16:17:20 -0300
description: On the implications of falling development cost # Add post description (optional)
img: office.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [generative AI, software, FOSS]
---


## Introduction

With recent advances in generative language modeling, it is becoming increasingly fruitful to apply these tools towards partially and fully automating software engineering tasks. As of mid-2024, it is common to hear the prediction that software engineering jobs will be lost to these technologies, and the software engineering jobs that remain will have a changed character. Having seen the utility of even a primitive generative autocomplete system, and the code generation quality of even some 7-billion-parameter models, I'm inclined to agree that there will be a significant impact. For the sake of this essay, let's assume the premise that advances in automatic software engineering tooling drive the cost of any purely programming software engineering task lower and lower over time. I think this is a reasonable general assumption to make, even if it doesn't hold true for every task. In this setting, I think free software will become more competitive with proprietary software across many previously defensible niches, and ultimately we should expect to see more free software and less proprietary software. The world is chaotic and hard to predict, and incumbents are resourceful and motivated to defend their incumbency, so perhaps it is better to take this more as a rallying cry than a prophecy.

## Why Should We Want Free Software to Replace Proprietary Software?

When someone says free software or open source software, often what they are referring to is the more strictly defined "free and open source software," which is software that doesn't restrict how the software is used and either doesn't restrict how software is distributed at all, or at most imposes the requirement that subsequently republished derivatives must also be free and open source. This software has several structural advantages over proprietary software.

### Availability

Free software is guaranteed to remain available and usable to a user (though a user may need to maintain or host it themselves), since the license cannot be revoked, and the user is able to save the source code on their own systems. Also, free software is usable without any cost, making it available to a much wider range of users.

### Security

Closed source software cannot have its security verified. Period. Ultimately, if you run a compiled program on your machine, you are taking the word of the entity that compiled it that it is not malicious. Anyone who is actually serious about security must include the people that write their software and create their hardware in their threat model. A caveat is that proprietary, source-available software can gain this advantage without becoming free and open source.

### Alignment With User Goals

Because free software is open source, features not aligned with user goals, such as unblockable ads, spyware, or use restrictions, can be removed by users in derivative versions of the software. This creates pressure on maintainers to respect their users and means that generally, free software respects user freedom. Projects that lose the trust of their community risk losing their community to a derivative of their own software maintained by a competing team. A perfect example of free software protecting a tangible and important aspect of user alignment is how Google is unable to cripple UBlock Origin on Firefox.

While there are still challenges in many niches in bringing the quality of free software up to the quality of proprietary software, all free software already has these structural advantages over all proprietary software. Assuming all else equal, it is always better to use free software than proprietary software, as you have **more options and more assurances**.

## What Niches Are Currently Defensible by Proprietary Software?

Currently, most niches of proprietary software remain defensible. Computer-Aided Design (CAD), DAW (Digital Audio Workstation), image editing, video editing, document editing, desktop operating systems, and spreadsheet software are all still dominated by proprietary software. At current levels of programmer efficiency, proprietary software is likely to be able to defend these niches at least into the near future.

## The Software Cost Transition

It is worth considering what will happen if the cost of writing software gets really low. We can model this in a really naive way. Just assume the average programmer starts to become on average some factor above 1 more productive every year when aided by generative tooling. You can pick whatever factor you want, even a conservative one. If this happens, given the same amount of development, naively you should also see an improvement in the efficiency of development in all software projects, both open and closed source. So naively, both free software projects and proprietary software projects should both become better at a similar rate, with the proprietary projects always staying somewhat ahead due to superior funding.

But in reality, there is a ceiling for what features are useful in any particular type of software. It's hard to say for certain when a particular piece of software has saturated its useful features, but I think a good example of a niche that has already fallen under the waterline is version control. It would be a very dangerous bet to try to outcompete Git with any piece of proprietary software; even if you had 10 billion dollars to put into its development, you might still fail.

As open source projects increase in scale, they will have enough resources despite their small funding to push this waterline ever higher. To document editing, to image and audio editing, to CAD software, to DAW. There will simply be free software alternatives that are perfectly good, and even hugely more resourced companies will not be able to make software sufficiently superior to justify any sort of payment to or agreement to any terms from the vendor.

Companies can try to defend their niche by popularizing proprietary data formats and familiarizing professionals with their proprietary software-specific interfaces and toolsets, but ultimately, these are all shallow moats, and none can be trusted to hold the rising waterline indefinitely. Proprietary software vendors should always try their best to provide whatever remaining features they can that are still missing from their open source competitors; in an ideal world, they will be driven to add value in the attempt to stay above the free and open source feature waterline until that waterline reaches the useful feature ceiling.

I think this is a likely future for many niches.

## What Niches Will Remain Defensible?

Some niches will remain defensible, and these include: software tied to patented or copyrighted hardware, branded software that acts as the on-device representative of a corporation (Panera app, Starbucks app, bank apps, etc.), video games, and software that relies on API calls where the API provides a service that can't be replicated with local inference (for example, very powerful AI inference).

## Closing Remarks

I don't think the future I described here is guaranteed, but readers should take away from this that if programmers are, in fact, becoming more and more efficient, it's a better time than ever to contribute to free software, as there is a real chance to compete with and ultimately displace proprietary software from many important niches.





## Feedback

As a final note, if you have corrections or want to add notes from your own experience, please add them as a PR on the source repository for this blog:

[https://github.com/balisujohn/balisujohn.github.io/](https://github.com/balisujohn/balisujohn.github.io/)
